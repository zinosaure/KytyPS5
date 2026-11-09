#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_

#include "common/common.h"
#include "common/commandThreadProfile.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/renderer/dynamicState.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/host_gpu/renderer/render.h"

#include <array>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace Libs::Graphics {

class Buffer;

class CommandScheduler {
public:
	CommandScheduler(RenderContext& context, GraphicContext& graphics);
	~CommandScheduler();
	KYTY_CLASS_NO_COPY(CommandScheduler);

	void           Begin(HW::Context& registers, HW::UserConfig& user_config, HW::Shader& shaders);
	void           EndRendering() { m_command.EndRendering(); }
	void           Flush(Common::CommandThreadProfile::Close reason = Common::CommandThreadProfile::Close::Other);
	void           Flush(SubmitInfo& submit);
	// A draw or a dispatch was recorded: the work goes to the GPU when it would otherwise run dry.
	void           Recorded();
	// The end of one of the game's command lists: submits unless the GPU has two buffers queued and this one is
	// not full yet. True when it submitted.
	bool           FlushAtBoundary();
	// Before the GPU thread sleeps: whatever is recorded goes to the GPU.
	void           FlushBeforeIdle();
	void           Finish();
	CommandBuffer& BeginCommand();
	uint64_t       Submit(SubmitInfo submit = {});
	// Deferred callbacks can observe an externally owned drain, but cannot initiate shutdown:
	// the priority runner cannot join itself.
	void                      Shutdown();
	// GameFlex: this scheduler feeds the performance overlay (CommandThreadProfile): why buffers close, and
	// how long the GPU takes over each one, from timestamps at its begin and end.
	void                      EnableProfile();
	void                      Wait(uint64_t tick);
	void                      PopPendingOperations();
	void                      DrainPriorityOperations();
	void                      WaitPriorityOperations(uint64_t tick);
	// Guest-memory completions use the priority queue; normal callbacks maintain GPU resources.
	void                      DeferOperation(Common::UniqueFunction<void>&& operation);
	void                      DeferPriorityOperation(Common::UniqueFunction<void>&& operation);
	[[nodiscard]] bool        HasPendingPriorityOperations();
	[[nodiscard]] static bool InDeferredOperation() noexcept;

	[[nodiscard]] bool Active() const noexcept { return m_command.m_registers != nullptr; }
	void                           CheckActive() const;
	CommandBuffer&                 Current();
	[[nodiscard]] uint64_t         CurrentTick() const noexcept { return m_master.CurrentTick(); }
	[[nodiscard]] bool             IsFree(uint64_t tick);
	[[nodiscard]] MasterSemaphore& GetMasterSemaphore() noexcept { return m_master; }
	[[nodiscard]] DynamicState&    GetDynamicState() noexcept { return m_dynamic_state; }
	[[nodiscard]] RenderContext&   Context() const noexcept { return m_context; }
	[[nodiscard]] GraphicContext&  Graphics() const noexcept { return m_graphics; }

private:
	class CommandPool {
	public:
		CommandPool(GraphicContext& graphics, MasterSemaphore& master);
		~CommandPool();
		KYTY_CLASS_NO_COPY(CommandPool);

		vk::CommandBuffer Commit();

	private:
		static constexpr size_t GrowStep = 4;

		size_t Grow();

		GraphicContext&                m_graphics;
		MasterSemaphore&               m_master;
		vk::CommandPool                m_pool = nullptr;
		std::vector<vk::CommandBuffer> m_buffers;
		std::vector<uint64_t>          m_ticks;
		size_t                         m_hint = 0;
	};

	enum class OperationState { Open, Draining, Closed };

	struct PendingOperation {
		Common::UniqueFunction<void> callback;
		uint64_t                     tick = 0;
	};

	void BeginNext();
	void HarvestTimestamps();
	void PriorityOperationsThread(std::stop_token stop);
	void QueueOperation(Common::UniqueFunction<void>&& operation, bool priority);
	void RunOperation(Common::UniqueFunction<void>&& operation);

	MasterSemaphore              m_master;
	RenderContext&               m_context;
	GraphicContext&              m_graphics;
	CommandPool                  m_command_pool;
	CommandBuffer                m_command;
	DynamicState                 m_dynamic_state;
	std::queue<PendingOperation> m_pending_operations;
	std::queue<PendingOperation> m_priority_operations;
	std::mutex                   m_operation_mutex;
	// GameFlex: PopPendingOperations calls since the driver was last asked where the GPU is.
	uint32_t m_refresh_skips = 0;
	// The draws and dispatches recorded in the current buffer, and whether anything at all was (Current() hands
	// the buffer to whoever records: a label, a transfer, a draw).
	uint32_t m_commands_recorded = 0;
	bool     m_buffer_used       = false;
	// The word the GPU writes at the end of each command buffer with the buffer's tick (VK_AMD_buffer_marker):
	// where the GPU is, read from memory (MasterSemaphore::Progress).
	std::unique_ptr<Buffer> m_progress;
	std::condition_variable      m_operation_available;
	std::jthread                 m_priority_thread;
	uint64_t                     m_priority_active_tick = 0; // 0 while none runs
	OperationState               m_operation_state      = OperationState::Open;
	// GameFlex: the instrument (EnableProfile).
	static constexpr uint32_t                 kTimestampSlots = 256;
	bool                                      m_profiled      = false;
	Common::CommandThreadProfile::Close       m_close_reason  = Common::CommandThreadProfile::Close::Other;
	vk::QueryPool                             m_timestamps    = nullptr;
	double                                    m_timestamp_ns  = 1.0;
	uint64_t                                  m_timestamp_mask = ~uint64_t {0};
	std::array<uint64_t, kTimestampSlots>     m_timestamp_ticks {};
	std::array<bool, kTimestampSlots>         m_timestamp_starved {}; // The GPU was dry when it was submitted.
	std::deque<uint64_t>                      m_timestamp_pending;
	uint64_t                                  m_last_end_tick = 0;
	uint64_t                                  m_last_end      = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
