#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "common/commandThreadProfile.h"
#include "common/gameflexWatchdog.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fmt/format.h>
#include <optional>
#include <string>
#include <unwind.h>
#include <vector>

namespace Libs::Graphics {

static thread_local CommandScheduler* g_deferred_callback_scheduler = nullptr;

CommandScheduler::CommandPool::CommandPool(GraphicContext& graphics, MasterSemaphore& master)
    : m_graphics(graphics), m_master(master) {
	EXIT_IF(graphics.queue_family == static_cast<uint32_t>(-1));
	vk::CommandPoolCreateInfo create {};
	create.queueFamilyIndex = graphics.queue_family;
	create.flags            = vk::CommandPoolCreateFlagBits::eTransient |
	                          vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
	const auto result       = graphics.device.createCommandPool(&create, nullptr, &m_pool);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_pool == nullptr);
}

CommandScheduler::CommandPool::~CommandPool() {
	m_graphics.device.destroyCommandPool(m_pool, nullptr);
}

size_t CommandScheduler::CommandPool::Grow() {
	const auto first = m_ticks.size();
	m_ticks.resize(first + GrowStep);
	m_buffers.resize(first + GrowStep);

	vk::CommandBufferAllocateInfo allocate {};
	allocate.commandPool        = m_pool;
	allocate.level              = vk::CommandBufferLevel::ePrimary;
	allocate.commandBufferCount = static_cast<uint32_t>(GrowStep);
	EXIT_IF(m_graphics.device.allocateCommandBuffers(&allocate, m_buffers.data() + first) !=
	        vk::Result::eSuccess);
	return first;
}

vk::CommandBuffer CommandScheduler::CommandPool::Commit() {
	auto       gpu_tick = m_master.KnownGpuTick();
	const auto search   = [this, &gpu_tick](size_t begin, size_t end) -> std::optional<size_t> {
		for (size_t index = begin; index < end; ++index) {
			if (gpu_tick >= m_ticks[index]) {
				m_ticks[index] = m_master.CurrentTick();
				return index;
			}
		}
		return std::nullopt;
	};

	auto found = search(m_hint, m_ticks.size());
	if (!found) {
		m_master.Refresh();
		gpu_tick = m_master.KnownGpuTick();
		found    = search(m_hint, m_ticks.size());
	}
	if (!found) {
		found = search(0, m_hint);
	}
	if (!found) {
		found           = Grow();
		m_ticks[*found] = m_master.CurrentTick();
	}

	m_hint = (*found + 1) % m_ticks.size();
	return m_buffers[*found];
}

bool CommandScheduler::InDeferredOperation() noexcept {
	return g_deferred_callback_scheduler != nullptr;
}

CommandScheduler::CommandScheduler(RenderContext& context, GraphicContext& graphics)
    : m_master(graphics), m_context(context), m_graphics(graphics),
      m_command_pool(graphics, m_master), m_command(*this),
      m_priority_thread([this](std::stop_token stop) { PriorityOperationsThread(stop); }) {
	if (graphics.buffer_marker_enabled) {
		m_progress = std::make_unique<Buffer>(graphics, *this, MemoryUsage::Download, 0,
		                                      vk::BufferUsageFlagBits::eTransferDst, 64);
		if (m_progress->IsCoherent() && !m_progress->Mapped().empty()) {
			std::memset(m_progress->Mapped().data(), 0, m_progress->Mapped().size());
			m_master.SetProgressWord(reinterpret_cast<const uint32_t*>(m_progress->Mapped().data()));
		} else {
			m_progress.reset();
		}
	}
}

CommandScheduler::~CommandScheduler() {
	Shutdown();
	if (m_timestamps != nullptr) {
		m_graphics.device.destroyQueryPool(m_timestamps, nullptr);
	}
}

void CommandScheduler::EnableProfile() {
	m_profiled         = true;
	const auto& limits = m_graphics.GetPhysicalDeviceProperties().limits;
	if (!m_graphics.host_query_reset_enabled || limits.timestampComputeAndGraphics == VK_FALSE) {
		return;
	}
	const vk::QueryPoolCreateInfo info {.queryType = vk::QueryType::eTimestamp, .queryCount = 2 * kTimestampSlots};
	if (m_graphics.device.createQueryPool(&info, nullptr, &m_timestamps) != vk::Result::eSuccess) {
		m_timestamps = nullptr;
		return;
	}
	m_graphics.device.resetQueryPool(m_timestamps, 0, 2 * kTimestampSlots);
	m_timestamp_ns     = static_cast<double>(limits.timestampPeriod);
	const auto queues = m_graphics.physical_device.getQueueFamilyProperties();
	if (m_graphics.queue_family < queues.size()) {
		const auto bits = queues[m_graphics.queue_family].timestampValidBits;
		m_timestamp_mask = bits >= 64 ? ~uint64_t {0} : (uint64_t {1} << bits) - 1;
	}
}

// The GPU's begin and end of each buffer it has finished: its busy time, and the gap since the previous one.
void CommandScheduler::HarvestTimestamps() {
	uint64_t   busy    = 0;
	uint64_t   gap     = 0;
	uint64_t   queued  = 0;
	uint64_t   buffers = 0;
	const auto done    = m_master.KnownGpuTick();
	while (!m_timestamp_pending.empty() && m_timestamp_pending.front() <= done) {
		const auto tick = m_timestamp_pending.front();
		m_timestamp_pending.pop_front();
		const auto slot = static_cast<uint32_t>(tick % kTimestampSlots);
		if (m_timestamp_ticks[slot] != tick) {
			continue;
		}
		std::array<uint64_t, 4> data {};
		const auto result = m_graphics.device.getQueryPoolResults(
		    m_timestamps, 2 * slot, 2, sizeof(data), data.data(), 2 * sizeof(uint64_t),
		    vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWithAvailability);
		if ((result != vk::Result::eSuccess && result != vk::Result::eNotReady) || data[1] == 0 || data[3] == 0) {
			continue;
		}
		const auto begin = data[0] & m_timestamp_mask;
		const auto end   = data[2] & m_timestamp_mask;
		if (end >= begin) {
			busy += end - begin;
			buffers++;
		}
		if (m_last_end_tick + 1 == tick && begin > m_last_end) {
			(m_timestamp_starved[slot] ? gap : queued) += begin - m_last_end;
		}
		m_last_end_tick = tick;
		m_last_end      = end;
	}
	if (buffers != 0) {
		Common::CommandThreadProfile::AddGpuTimes(static_cast<uint64_t>(static_cast<double>(busy) * m_timestamp_ns),
		                                          static_cast<uint64_t>(static_cast<double>(gap) * m_timestamp_ns),
		                                          static_cast<uint64_t>(static_cast<double>(queued) * m_timestamp_ns),
		                                          buffers);
	}
}

void CommandScheduler::Shutdown() {
	{
		std::unique_lock lock(m_operation_mutex);
		if (m_operation_state == OperationState::Closed) {
			return;
		}
		if (g_deferred_callback_scheduler == this) {
			EXIT_IF(m_operation_state == OperationState::Open);
			// A priority callback cannot join its own runner, while a normal callback can be
			// executing inside the shutdown owner's final PopPendingOperations. The owning
			// thread will finish shutdown after this callback returns.
			return;
		}
		if (m_operation_state == OperationState::Draining) {
			m_operation_available.wait(
			    lock, [this] { return m_operation_state == OperationState::Closed; });
			return;
		}
		m_operation_state = OperationState::Draining;
	}
	if (!m_command.IsInvalid()) {
		Submit();
	}
	m_master.Wait(CurrentTick() - 1);
	PopPendingOperations();
	DrainPriorityOperations();
	m_priority_thread.request_stop();
	m_operation_available.notify_all();
	if (m_priority_thread.joinable()) {
		m_priority_thread.join();
	}
	{
		std::lock_guard lock(m_operation_mutex);
		EXIT_IF(!m_pending_operations.empty() || !m_priority_operations.empty() ||
		        m_priority_active_tick != 0);
		m_operation_state = OperationState::Closed;
	}
	m_operation_available.notify_all();
}

void CommandScheduler::Begin(HW::Context& registers, HW::UserConfig& user_config,
                             HW::Shader& shaders) {
	{
		std::lock_guard lock(m_operation_mutex);
		EXIT_IF(m_operation_state != OperationState::Open);
	}
	m_command.Bind(registers, user_config, shaders);

	if (m_command.IsInvalid()) {
		BeginNext();
	}
}

void CommandScheduler::Flush(Common::CommandThreadProfile::Close reason) {
	m_close_reason = reason;
	SubmitInfo submit;
	Flush(submit);
}

// The submission follows the GPU. The console's command processor consumes the ring as the game fills it; the
// host GPU only sees what vkQueueSubmit hands it. Every submission costs the GPU a drain and a cold start, and
// the thread a system call plus the driver's walk of every buffer the submission references: about 0.5 ms here
// (vkQueueSubmit, 17 % of the command thread in Astro Bot at 52 submissions an image). Every handoff between two
// buffers costs the GPU 0.6 ms when the next one is not in its queue yet. So: a small buffer only when the GPU has
// run dry, a full one while it has work queued, and at a boundary of the game's lists a middling one unless the GPU
// still has a buffer behind the current one; the fixed points (suspension, flip, waits) submit as before.
namespace {
constexpr uint32_t kCheckCommands    = 64;   // how often Recorded() looks at the GPU
constexpr uint32_t kBoundaryCommands = 256;  // enough to submit at a list boundary with one buffer in flight
constexpr uint32_t kFullCommands     = 2048; // a full buffer
} // namespace

void CommandScheduler::Recorded() {
	if (m_command.IsInvalid()) {
		return;
	}
	m_commands_recorded++;
	if (m_commands_recorded % kCheckCommands != 0) {
		return;
	}
	// Buffers submitted and not finished, as the GPU says now. None: the GPU is dry, this one goes now.
	const uint64_t queued = CurrentTick() - 1 - m_master.Progress();
	if (queued == 0 || m_commands_recorded >= kFullCommands) {
		Flush(queued == 0 ? Common::CommandThreadProfile::Close::GpuIdle : Common::CommandThreadProfile::Close::Size);
	}
}

bool CommandScheduler::FlushAtBoundary() {
	if (!Active() || m_command.IsInvalid() || !m_buffer_used) {
		return false;
	}
	// Dry: now. One buffer in flight: now if there is enough here for the GPU to find it in its queue when that
	// one ends, else the next list joins. Two or more: the next list joins, until the buffer is full.
	const uint64_t queued = CurrentTick() - 1 - m_master.Progress();
	if (queued == 0 || (queued == 1 && m_commands_recorded >= kBoundaryCommands) ||
	    m_commands_recorded >= kFullCommands) {
		Flush(m_commands_recorded >= kFullCommands ? Common::CommandThreadProfile::Close::Size : Common::CommandThreadProfile::Close::Boundary);
		return true;
	}
	return false;
}

void CommandScheduler::FlushBeforeIdle() {
	// Before the first Begin, or after a reset, nothing can have been recorded.
	if (Active() && !m_command.IsInvalid() && m_buffer_used) {
		Flush(Common::CommandThreadProfile::Close::BeforeIdle);
	}
}

void CommandScheduler::Flush(SubmitInfo& submit) {
	Submit(submit);
	BeginNext();
}

void CommandScheduler::Finish() {
	CheckActive();
	if (!m_command.IsInvalid()) {
		m_close_reason = Common::CommandThreadProfile::Close::Wait;
		Submit();
	}
	m_master.Wait(CurrentTick() - 1);
	BeginNext();
	PopPendingOperations();
}

void CommandScheduler::Wait(uint64_t tick) {
	Common::GameFlex::WaitNote gameflex_wait("host gpu: wait for a submission", tick);
	EXIT_IF(tick > CurrentTick());
	if (tick == CurrentTick()) {
		CheckActive();
		// A stream-buffer wrap can wait while a draw is being prepared through a reference to
		// Current(). The wrapper stays stable while its pooled Vulkan buffer is retired. Deferred
		// resources are released only at the next GPU operation boundary.
		m_close_reason            = Common::CommandThreadProfile::Close::Wait;
		const auto submitted_tick = Submit();
		EXIT_IF(submitted_tick != tick);
		m_master.Wait(tick);
		BeginNext();
	} else {
		m_master.Wait(tick);
	}
}

void CommandScheduler::PopPendingOperations() {
	// GameFlex: this runs at every draw, and asking the driver where the GPU is costs a system call
	// (vkGetSemaphoreCounterValue, an ioctl): 9 % of the command thread in Kingdom Hearts III. Ask only
	// when the oldest pending operation is not already known to be done, and when the word the GPU writes
	// at the end of each command buffer (0064) says it is: with the GPU behind, as in Astro Bot's garden,
	// asking once every 16 draws still cost 1.6 % of the thread for nothing. The semaphore decides. Without
	// that word, once every 16 calls; these operations wait for the GPU anyway.
	{
		std::lock_guard lock(m_operation_mutex);
		if (m_pending_operations.empty()) {
			return;
		}
		const auto tick = m_pending_operations.front().tick;
		if (!m_master.IsFree(tick) && (m_master.HasProgressWord() ? m_master.Progress() < tick
		                                                          : (++m_refresh_skips & 15u) != 0)) {
			return;
		}
	}
	m_master.Refresh();
	for (;;) {
		PendingOperation operation;
		{
			std::lock_guard lock(m_operation_mutex);
			if (m_pending_operations.empty() ||
			    !m_master.IsFree(m_pending_operations.front().tick)) {
				return;
			}
			operation = std::move(m_pending_operations.front());
			m_pending_operations.pop();
		}
		WaitPriorityOperations(operation.tick);
		RunOperation(std::move(operation.callback));
	}
}

void CommandScheduler::DeferOperation(Common::UniqueFunction<void>&& operation) {
	QueueOperation(std::move(operation), false);
}

void CommandScheduler::DeferPriorityOperation(Common::UniqueFunction<void>&& operation) {
	QueueOperation(std::move(operation), true);
}

bool CommandScheduler::DeferPriorityOperationOnSubmitted(Common::UniqueFunction<void>& operation) {
	CheckActive();
	EXIT_IF(!operation);
	if (m_buffer_used || m_command.IsInvalid()) {
		return false;
	}
	std::unique_lock lock(m_operation_mutex);
	const auto       tick = CurrentTick();
	if (m_operation_state != OperationState::Open || m_operation_tick == tick) {
		return false;
	}
	// The queue stays in tick order: no operation waits for the open buffer, so none is past tick - 1.
	m_priority_operations.push({std::move(operation), tick - 1});
	lock.unlock();
	m_operation_available.notify_one();
	return true;
}

void CommandScheduler::QueueOperation(Common::UniqueFunction<void>&& operation, bool priority) {
	CheckActive();
	EXIT_IF(!operation);
	std::unique_lock lock(m_operation_mutex);
	if (m_operation_state == OperationState::Open) {
		auto&      queue = priority ? m_priority_operations : m_pending_operations;
		const auto tick  = CurrentTick();
		queue.push({std::move(operation), tick});
		m_operation_tick = tick;
		lock.unlock();
		if (priority) {
			m_operation_available.notify_one();
		}
		return;
	}
	if (g_deferred_callback_scheduler != this) {
		m_operation_available.wait(lock,
		                           [this] { return m_operation_state == OperationState::Closed; });
	}
	lock.unlock();
	operation();
}

void CommandScheduler::NoteCloseOperations() {
	std::lock_guard lock(m_operation_mutex);
	m_close_operations = m_operation_tick == CurrentTick();
}

bool CommandScheduler::HasPendingPriorityOperations() {
	std::lock_guard lock(m_operation_mutex);
	return !m_priority_operations.empty() || m_priority_active_tick != 0;
}

void CommandScheduler::PriorityOperationsThread(std::stop_token stop) {
	while (!stop.stop_requested()) {
		PendingOperation operation;
		{
			std::unique_lock lock(m_operation_mutex);
			m_operation_available.wait(lock, [this, &stop] {
				return stop.stop_requested() || !m_priority_operations.empty();
			});
			if (stop.stop_requested()) {
				return;
			}
			operation = std::move(m_priority_operations.front());
			m_priority_operations.pop();
			m_priority_active_tick = operation.tick;
		}
		m_master.Wait(operation.tick);
		if (!stop.stop_requested()) {
			RunOperation(std::move(operation.callback));
		}
		{
			std::lock_guard lock(m_operation_mutex);
			m_priority_active_tick = 0;
		}
		m_operation_available.notify_all();
	}
}

void CommandScheduler::DrainPriorityOperations() {
	EXIT_IF(g_deferred_callback_scheduler == this);
	std::unique_lock lock(m_operation_mutex);
	m_operation_available.wait(
	    lock, [this] { return m_priority_operations.empty() && m_priority_active_tick == 0; });
}

void CommandScheduler::WaitPriorityOperations(uint64_t tick) {
	EXIT_IF(g_deferred_callback_scheduler == this);
	Common::CommandThreadProfile::Scope ctp_wait(Common::CommandThreadProfile::Cause::WaitGpuPriority);
	std::unique_lock lock(m_operation_mutex);
	m_operation_available.wait(lock, [this, tick] {
		const bool active_before_or_at =
		    m_priority_active_tick != 0 && m_priority_active_tick <= tick;
		const bool queued_before_or_at =
		    !m_priority_operations.empty() && m_priority_operations.front().tick <= tick;
		return !active_before_or_at && !queued_before_or_at;
	});
}

void CommandScheduler::RunOperation(Common::UniqueFunction<void>&& operation) {
	auto* previous                = g_deferred_callback_scheduler;
	g_deferred_callback_scheduler = this;
	operation();
	g_deferred_callback_scheduler = previous;
}

bool CommandScheduler::IsFree(uint64_t tick) {
	if (m_master.IsFree(tick)) {
		return true;
	}
	m_master.Refresh();
	return m_master.IsFree(tick);
}

namespace {

_Unwind_Reason_Code CollectFrame(_Unwind_Context* context, void* frames) {
	auto* list = static_cast<std::vector<uintptr_t>*>(frames);
	if (list->size() >= 32) {
		return _URC_END_OF_STACK;
	}
	list->push_back(static_cast<uintptr_t>(_Unwind_GetIP(context)));
	return _URC_NO_REASON;
}

} // namespace

void CommandScheduler::CheckActive() const {
	if (!Active()) {
		// The fatal error below names only this line. The return addresses say which caller
		// used the scheduler before its first Begin (addr2line on the build, minus its base).
		std::vector<uintptr_t> frames;
		_Unwind_Backtrace(CollectFrame, &frames);
		std::string text = "CommandScheduler used before Begin; backtrace:";
		for (const auto ip: frames) {
			text += fmt::format(" {:#x}", ip);
		}
		Log::WriteToConsoleAndLog(text + "\n");
	}
	EXIT_IF(!Active());
}

CommandBuffer& CommandScheduler::Current() {
	EXIT_IF(m_command.IsInvalid());
	m_buffer_used = true;
	return m_command;
}

CommandBuffer& CommandScheduler::BeginCommand() {
	Common::CommandThreadProfile::Scope ctp_submit(Common::CommandThreadProfile::Cause::SubmitBegin);
	EXIT_IF(!m_command.IsInvalid());
	m_command.m_buffer = m_command_pool.Commit();
	m_command.Begin();
	m_dynamic_state.Invalidate();
	if (m_profiled && Common::DebugCounters::Enabled()) {
		Common::CommandThreadProfile::Count(Common::CommandThreadProfile::Event::CommandBuffers);
		if (m_timestamps != nullptr) {
			const auto ctp_timestamps = Common::CommandThreadProfile::Clock();
			HarvestTimestamps();
			// This buffer's tick is the next one handed out; its slot was last used 256 buffers ago.
			const auto tick = CurrentTick();
			const auto slot = static_cast<uint32_t>(tick % kTimestampSlots);
			if (m_timestamp_ticks[slot] == 0 || m_master.IsFree(m_timestamp_ticks[slot])) {
				m_graphics.device.resetQueryPool(m_timestamps, 2 * slot, 2);
				m_command.Handle().writeTimestamp2(vk::PipelineStageFlagBits2::eNone, m_timestamps, 2 * slot);
				m_timestamp_ticks[slot] = tick;
			}
			Common::CommandThreadProfile::AddTimestampTicks(ctp_timestamps);
		}
	}
	return m_command;
}

uint64_t CommandScheduler::Submit(SubmitInfo submit) {
	Common::CommandThreadProfile::Scope ctp_submit_scope(Common::CommandThreadProfile::Cause::SubmitEnd);
	EXIT_IF(m_command.IsInvalid());

	// This thread alone hands out ticks and submits: the order of the two is the same.
	const auto tick = m_master.NextTick();
	if (m_progress != nullptr) {
		// Where the GPU is, for the CPU to read: this buffer's tick, written once all its work is done.
		m_command.EndRendering();
		m_command.Handle().writeBufferMarker2AMD(vk::PipelineStageFlagBits2::eAllCommands,
		                                          m_progress->Handle(), 0, static_cast<uint32_t>(tick));
	}
	if (const auto slot = static_cast<uint32_t>(tick % kTimestampSlots);
	    m_timestamps != nullptr && m_timestamp_ticks[slot] == tick) {
		const auto ctp_timestamps = Common::CommandThreadProfile::Clock();
		m_command.EndRendering();
		m_command.Handle().writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, m_timestamps, 2 * slot + 1);
		m_timestamp_pending.push_back(tick);
		m_timestamp_starved[slot] = tick == m_master.Progress() + 1;
		Common::CommandThreadProfile::AddTimestampTicks(ctp_timestamps);
	}
	m_command.End();
	const auto buffer   = m_command.m_buffer;
	auto&      graphics = m_graphics;
	EXIT_IF(graphics.queue == nullptr);

	vk::Result result;
	{
		Common::CommandThreadProfile::Phase(Common::CommandThreadProfile::Cause::WaitQueueLock);
		Common::LockGuard lock(graphics.queue_mutex);
		Common::CommandThreadProfile::Phase(Common::CommandThreadProfile::Cause::SubmitQueue);
		submit.AddSignal(m_master.Handle(), tick);

		vk::TimelineSemaphoreSubmitInfo timeline_info {};
		timeline_info.waitSemaphoreValueCount   = submit.num_wait_semaphores;
		timeline_info.pWaitSemaphoreValues      = submit.wait_ticks.data();
		timeline_info.signalSemaphoreValueCount = submit.num_signal_semaphores;
		timeline_info.pSignalSemaphoreValues    = submit.signal_ticks.data();

		vk::SubmitInfo submit_info {};
		submit_info.pNext                = &timeline_info;
		submit_info.waitSemaphoreCount   = submit.num_wait_semaphores;
		submit_info.pWaitSemaphores      = submit.wait_semaphores.data();
		submit_info.pWaitDstStageMask    = submit.wait_stages.data();
		submit_info.commandBufferCount   = 1;
		submit_info.pCommandBuffers      = &buffer;
		submit_info.signalSemaphoreCount = submit.num_signal_semaphores;
		submit_info.pSignalSemaphores    = submit.signal_semaphores.data();

		result = graphics.queue.submit(1, &submit_info, nullptr);
	}
	Common::CommandThreadProfile::Phase(Common::CommandThreadProfile::Cause::SubmitEnd);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::QueueSubmits);

	if (result != vk::Result::eSuccess) {
		EXIT("vkQueueSubmit failed: %s, tick=%" PRIu64 "\n", vk::to_string(result).c_str(), tick);
	}

	if (m_profiled) {
		// GameFlex: an empty buffer, and whether a deferred operation waits for its tick all the same.
		bool operations = false;
		if (!m_buffer_used && Common::DebugCounters::Enabled()) {
			if (m_close_operations.has_value()) {
				operations = *m_close_operations;
			} else {
				std::lock_guard lock(m_operation_mutex);
				operations = m_operation_tick == tick;
			}
		}
		Common::CommandThreadProfile::CountClose(m_close_reason, !m_buffer_used, operations);
		Common::CommandThreadProfile::Count(Common::CommandThreadProfile::Event::Submits);
	}
	m_close_reason      = Common::CommandThreadProfile::Close::Other;
	m_close_operations.reset();
	m_command.m_buffer  = nullptr;
	m_commands_recorded = 0;
	m_buffer_used       = false;
	return tick;
}

void CommandScheduler::BeginNext() {
	CheckActive();
	BeginCommand();
}

} // namespace Libs::Graphics
