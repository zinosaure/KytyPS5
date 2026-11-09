#ifndef KYTY_COMMON_COMMAND_THREAD_PROFILE_H_
#define KYTY_COMMON_COMMAND_THREAD_PROFILE_H_

#include "common/debugCounters.h"
#include "common/timer.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <utility>

#if defined(_M_X64) || defined(__x86_64__)
#include <x86intrin.h>
#endif

// GameFlex: where the GPU command thread spends its time. Every moment of that thread belongs to
// exactly one cause: entering a cause suspends the current one, leaving it resumes the outer one,
// so the causes add up to the thread's wall time - its work and its waits. Collected only while the
// performance overlay (or --perf-log) asks for counters, like DebugCounters.
namespace Common::CommandThreadProfile {

enum class Cause : uint8_t {
	// Work.
	Pm4,       // The PM4 loop and the packets listed nowhere below (sync, labels, markers).
	Registers, // Register writes.
	Draw,      // Draw and dispatch packets, outside the steps below.
	Programs,  // Shader programs and pipelines found for a draw or dispatch.
	Resources, // Resource tables (SRT) walked and materialized.
	Textures,  // Images and views found or created for bindings.
	Buffers,   // Buffers found or created for bindings.
	Targets,   // Render targets and depth.
	Bindings,  // Descriptors written and bound.
	Barriers,  // Barriers recorded (the guest's synchronization, image layouts).
	Record,    // Vulkan recording of a draw or dispatch: pipeline, state, the call itself.
	Compile,   // Shaders translated, pipelines created.
	Transfers, // DMA copies and fills.
	Submit,    // Command buffers begun, ended and submitted.
	Upkeep,    // Frame boundaries: suspend points, flips, cache aging.
	Foreign,   // Work other threads hand to this one.
	// Waits.
	WaitGpuRead,     // For GPU writes the CPU must read (indirect counts, CP reads, readbacks).
	WaitGpuSrt,      // For GPU writes to a resource table being read.
	WaitGpuRebind,   // For the GPU before unified memory is remapped.
	WaitGpuFrame,    // For presentation: a free frame, room in the flip queue, a flip done.
	WaitGpuPriority, // For the thread that delivers the GPU's completions to the guest.
	WaitGpuOther,    // For the GPU, for another reason.
	WaitQueueLock,   // For the Vulkan queue, held by presentation.
	WaitLabelGpu,    // Every guest queue waits for a value a write still on its way to the GPU will set.
	WaitLabelCpu,    // Every guest queue waits for a value nothing on the GPU will set (the game, another queue).
	IdleSuspend,     // No guest work: the game waits at its suspend point for the previous frame.
	Idle,            // No guest work.
	Other,           // Not attributed.
	Count
};

// Why the command thread is about to wait for the GPU, set by the callers of the waits.
enum class WaitReason : uint8_t { Other, Read, Srt, Rebind, Frame };

constexpr size_t CAUSE_COUNT = static_cast<size_t>(Cause::Count);

enum class Event : uint8_t {
	Packets,
	Draws,
	Dispatches,
	Barriers,
	CommandBuffers,
	Submits,
	BlockedRetries, // Guest command lists resumed while still waiting.
	LabelPolls,     // Wake-ups while every queue waits for a label.
	LabelWaitsGpu,  // Guest waits suspended on a value a pending GPU write will set.
	LabelWaitsCpu,  // Guest waits suspended on a value nothing pending on the GPU will set.
	GpuWaits,       // Waits of this thread for the GPU.
	MemoryAllocations, // Vulkan device memory allocated (driver buffer objects), any thread.
	MemoryReleases,
	Count
};

constexpr size_t EVENT_COUNT = static_cast<size_t>(Event::Count);

// Why a command buffer was submitted.
enum class Close : uint8_t {
	GpuIdle,    // The GPU ran dry while commands were recorded.
	Size,       // The buffer reached its command budget.
	Boundary,   // The end of a guest command list.
	BeforeIdle, // The thread was about to sleep or poll.
	Wait,       // The CPU needed the GPU to finish (readback, Finish).
	Frame,      // A suspend point or a flip.
	Other,
	Count
};

constexpr size_t CLOSE_COUNT  = static_cast<size_t>(Close::Count);
constexpr size_t PACKET_KINDS = 256; // PM4 type-3 opcodes.

namespace Detail {
inline std::atomic<uint64_t> g_ticks[CAUSE_COUNT] {};
inline std::atomic<uint64_t> g_events[EVENT_COUNT] {};
inline std::atomic<uint64_t> g_closes[CLOSE_COUNT] {};
inline std::atomic<uint64_t> g_packets[PACKET_KINDS] {};
// From per-command-buffer GPU timestamps: busy time, gaps between consecutive buffers, buffers measured.
inline std::atomic<uint64_t> g_gpu_busy_ns {0};
inline std::atomic<uint64_t> g_gpu_gap_ns {0};      // The GPU ran dry: the buffer after the gap came late.
inline std::atomic<uint64_t> g_gpu_queued_gap_ns {0}; // The buffer after the gap was already waiting.
inline std::atomic<uint64_t> g_gpu_buffers {0};

struct ThreadState {
	bool     command_thread = false;
	bool     counting       = false; // The overlay was on at the last transition.
	Cause      current        = Cause::Other;
	WaitReason reason         = WaitReason::Other;
	bool       label_gpu      = true; // What the last suspended guest wait waits for.
	uint64_t   since          = 0;
};
inline thread_local ThreadState t_state;

[[nodiscard]] inline uint64_t Ticks() noexcept {
#if defined(_M_X64) || defined(__x86_64__)
	return __rdtsc();
#else
	return SteadyNs();
#endif
}

inline void Switch(Cause next) noexcept {
	auto& state = t_state;
	if (!state.command_thread) {
		return;
	}
	const bool counting = DebugCounters::Enabled();
	if (counting) {
		const auto now = Ticks();
		// The first transition after the overlay turns on only starts the clock.
		if (state.counting) {
			// One writer, this thread: a plain add, readers see a recent value.
			auto& ticks = g_ticks[static_cast<size_t>(state.current)];
			ticks.store(ticks.load(std::memory_order_relaxed) + (now - state.since), std::memory_order_relaxed);
		}
		state.since = now;
	}
	state.counting = counting;
	state.current  = next;
}
} // namespace Detail

// Called once by the command thread, before its first command.
inline void SetCommandThread() noexcept {
	Detail::t_state.command_thread = true;
	Detail::t_state.current        = Cause::Other;
}

[[nodiscard]] inline bool OnCommandThread() noexcept {
	return Detail::t_state.command_thread;
}

// The cause for the scope's lifetime; the outer cause resumes when it ends.
class Scope final {
public:
	explicit Scope(Cause cause) noexcept: m_outer(Detail::t_state.current) {
		Detail::Switch(cause);
	}
	~Scope() { Detail::Switch(m_outer); }
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;

private:
	Cause m_outer;
};

// Changes the current cause without restoring it: for the PM4 loop, inside a Scope that restores.
inline void Phase(Cause cause) noexcept {
	if (Detail::t_state.current != cause) {
		Detail::Switch(cause);
	}
}

// The reason for any wait for the GPU under it, on the command thread.
class WaitReasonScope final {
public:
	explicit WaitReasonScope(WaitReason reason) noexcept: m_outer(Detail::t_state.reason) {
		// The outermost reason wins: a readback inside a remap is the remap's.
		if (m_outer == WaitReason::Other) {
			Detail::t_state.reason = reason;
		}
	}
	~WaitReasonScope() { Detail::t_state.reason = m_outer; }
	WaitReasonScope(const WaitReasonScope&)            = delete;
	WaitReasonScope& operator=(const WaitReasonScope&) = delete;

private:
	WaitReason m_outer;
};

// The cause a wait for the GPU counts under, from the current reason.
[[nodiscard]] inline Cause WaitCause() noexcept {
	switch (Detail::t_state.reason) {
		case WaitReason::Read: return Cause::WaitGpuRead;
		case WaitReason::Srt: return Cause::WaitGpuSrt;
		case WaitReason::Rebind: return Cause::WaitGpuRebind;
		case WaitReason::Frame: return Cause::WaitGpuFrame;
		default: return Cause::WaitGpuOther;
	}
}

inline void AddGpuTimes(uint64_t busy_ns, uint64_t gap_ns, uint64_t queued_gap_ns, uint64_t buffers) noexcept {
	Detail::g_gpu_busy_ns.fetch_add(busy_ns, std::memory_order_relaxed);
	Detail::g_gpu_gap_ns.fetch_add(gap_ns, std::memory_order_relaxed);
	Detail::g_gpu_queued_gap_ns.fetch_add(queued_gap_ns, std::memory_order_relaxed);
	Detail::g_gpu_buffers.fetch_add(buffers, std::memory_order_relaxed);
}

inline void Count(Event event, uint64_t value = 1) noexcept {
	if (DebugCounters::Enabled()) {
		Detail::g_events[static_cast<size_t>(event)].fetch_add(value, std::memory_order_relaxed);
	}
}

// A guest wait suspends: on a value a pending GPU write will set, or not.
inline void NoteLabelWait(bool gpu) noexcept {
	Detail::t_state.label_gpu = gpu;
	Count(gpu ? Event::LabelWaitsGpu : Event::LabelWaitsCpu);
}

// The cause of a poll while every guest queue waits.
[[nodiscard]] inline Cause LabelCause() noexcept {
	return Detail::t_state.label_gpu ? Cause::WaitLabelGpu : Cause::WaitLabelCpu;
}

inline void CountClose(Close reason) noexcept {
	if (DebugCounters::Enabled()) {
		Detail::g_closes[static_cast<size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
	}
}

inline void CountPacket(uint32_t opcode) noexcept {
	if (DebugCounters::Enabled()) {
		Detail::g_packets[opcode & (PACKET_KINDS - 1)].fetch_add(1, std::memory_order_relaxed);
		Detail::g_events[static_cast<size_t>(Event::Packets)].fetch_add(1, std::memory_order_relaxed);
	}
}

struct Totals {
	std::array<uint64_t, CAUSE_COUNT>  ticks {};
	std::array<uint64_t, EVENT_COUNT>  events {};
	std::array<uint64_t, CLOSE_COUNT>  closes {};
	std::array<uint64_t, PACKET_KINDS> packets {};
	uint64_t                           gpu_busy_ns = 0;
	uint64_t                           gpu_gap_ns  = 0;
	uint64_t                           gpu_queued_gap_ns = 0;
	uint64_t                           gpu_buffers = 0;
	uint64_t                           clock_ticks = 0; // Ticks() when read.
	uint64_t                           clock_ns    = 0; // SteadyNs() when read: ticks per ns.
};

[[nodiscard]] inline Totals Read() noexcept {
	Totals totals;
	for (size_t i = 0; i < CAUSE_COUNT; i++) {
		totals.ticks[i] = Detail::g_ticks[i].load(std::memory_order_relaxed);
	}
	for (size_t i = 0; i < EVENT_COUNT; i++) {
		totals.events[i] = Detail::g_events[i].load(std::memory_order_relaxed);
	}
	for (size_t i = 0; i < CLOSE_COUNT; i++) {
		totals.closes[i] = Detail::g_closes[i].load(std::memory_order_relaxed);
	}
	for (size_t i = 0; i < PACKET_KINDS; i++) {
		totals.packets[i] = Detail::g_packets[i].load(std::memory_order_relaxed);
	}
	totals.gpu_busy_ns = Detail::g_gpu_busy_ns.load(std::memory_order_relaxed);
	totals.gpu_gap_ns  = Detail::g_gpu_gap_ns.load(std::memory_order_relaxed);
	totals.gpu_queued_gap_ns = Detail::g_gpu_queued_gap_ns.load(std::memory_order_relaxed);
	totals.gpu_buffers = Detail::g_gpu_buffers.load(std::memory_order_relaxed);
	totals.clock_ticks = Detail::Ticks();
	totals.clock_ns    = SteadyNs();
	return totals;
}

// Per-frame values between two readings, for the overlay and --perf-log.
struct Report {
	double                                     seconds = 0.0;
	double                                     frames  = 0.0;
	std::array<double, CAUSE_COUNT>            ms_per_frame {};
	std::array<double, EVENT_COUNT>            per_frame {};
	std::array<double, CLOSE_COUNT>            closes_per_frame {};
	std::array<std::pair<uint32_t, double>, 8> packets_per_frame {}; // The most frequent opcodes.
	double                                     gpu_busy_ms_per_frame = 0.0;
	double                                     gpu_gap_ms_per_frame  = 0.0;
	double                                     gpu_queued_gap_ms_per_frame = 0.0;
	double                                     wait_ms_per_frame = 0.0; // Every wait cause together.
	double                                     gpu_buffers_per_frame = 0.0;
};

[[nodiscard]] inline Report MakeReport(const Totals& before, const Totals& after,
                                       uint64_t frames) noexcept {
	Report report;
	const auto ns    = static_cast<double>(after.clock_ns - before.clock_ns);
	const auto ticks = static_cast<double>(after.clock_ticks - before.clock_ticks);
	if (ns <= 0.0 || ticks <= 0.0) {
		return report;
	}
	report.seconds          = ns / 1e9;
	report.frames           = static_cast<double>(frames);
	const double per_frame  = frames != 0 ? 1.0 / static_cast<double>(frames) : 0.0;
	const double ms_per_tick = ns / ticks / 1e6;
	for (size_t i = 0; i < CAUSE_COUNT; i++) {
		report.ms_per_frame[i] =
		    static_cast<double>(after.ticks[i] - before.ticks[i]) * ms_per_tick * per_frame;
	}
	for (size_t i = 0; i < EVENT_COUNT; i++) {
		report.per_frame[i] = static_cast<double>(after.events[i] - before.events[i]) * per_frame;
	}
	for (size_t i = 0; i < CLOSE_COUNT; i++) {
		report.closes_per_frame[i] =
		    static_cast<double>(after.closes[i] - before.closes[i]) * per_frame;
	}
	for (uint32_t op = 0; op < PACKET_KINDS; op++) {
		const double value = static_cast<double>(after.packets[op] - before.packets[op]) * per_frame;
		auto&        least = *std::min_element(
            report.packets_per_frame.begin(), report.packets_per_frame.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
		if (value > least.second) {
			least = {op, value};
		}
	}
	std::sort(report.packets_per_frame.begin(), report.packets_per_frame.end(),
	          [](const auto& a, const auto& b) { return a.second > b.second; });
	report.gpu_busy_ms_per_frame = static_cast<double>(after.gpu_busy_ns - before.gpu_busy_ns) / 1e6 * per_frame;
	report.gpu_gap_ms_per_frame  = static_cast<double>(after.gpu_gap_ns - before.gpu_gap_ns) / 1e6 * per_frame;
	report.gpu_queued_gap_ms_per_frame =
	    static_cast<double>(after.gpu_queued_gap_ns - before.gpu_queued_gap_ns) / 1e6 * per_frame;
	for (size_t i = static_cast<size_t>(Cause::WaitGpuRead); i <= static_cast<size_t>(Cause::WaitLabelCpu); i++) {
		report.wait_ms_per_frame += report.ms_per_frame[i];
	}
	report.gpu_buffers_per_frame = static_cast<double>(after.gpu_buffers - before.gpu_buffers) * per_frame;
	return report;
}

[[nodiscard]] constexpr const char* Name(Cause cause) noexcept {
	constexpr const char* NAMES[CAUSE_COUNT] = {
	    "pm4",           "registers",         "draw",          "programs",
	    "resources",     "textures",          "buffers",       "targets",
	    "bindings",      "barriers",          "record",        "compile",
	    "transfers",     "submit",            "upkeep",        "foreign",
	    "wait GPU read", "wait GPU SRT",      "wait GPU remap", "wait GPU frame",
	    "wait priority", "wait GPU other",    "wait queue lock", "wait label GPU", "wait label CPU",
	    "idle (suspend)", "idle",             "other"};
	return NAMES[static_cast<size_t>(cause)];
}

[[nodiscard]] constexpr const char* Name(Event event) noexcept {
	constexpr const char* NAMES[EVENT_COUNT] = {
	    "packets", "draws",       "dispatches", "barriers",    "command buffers", "submits",
	    "retries", "label polls", "label waits GPU", "label waits CPU", "GPU waits", "allocations", "releases"};
	return NAMES[static_cast<size_t>(event)];
}

[[nodiscard]] constexpr const char* Name(Close close) noexcept {
	constexpr const char* NAMES[CLOSE_COUNT] = {"GPU idle", "size",  "boundary", "before idle",
	                                            "wait",     "frame", "other"};
	return NAMES[static_cast<size_t>(close)];
}

} // namespace Common::CommandThreadProfile

#endif /* KYTY_COMMON_COMMAND_THREAD_PROFILE_H_ */
