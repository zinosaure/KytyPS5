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
	SubmitBegin, // Command buffers taken from the pool and begun, the GPU timestamps of finished ones read.
	SubmitEnd,   // Command buffers ended: the progress marker, the barrier the game still owes (GuestSync).
	SubmitQueue, // vkQueueSubmit itself, under the queue lock.
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
	TrustedWaits,   // Guest waits passed behind a barrier: a shader recorded before writes the value.
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
	// The frame's fixed points.
	SuspendPoint, // The game's suspend point (sceAgcSuspendPoint).
	Flip,         // A flip in the graphics command stream.
	CpuFlip,      // A flip the game submitted from the CPU.
	WaitFlipDone, // The stream waits until a flip is done.
	Other,
	Count
};

constexpr size_t CLOSE_COUNT  = static_cast<size_t>(Close::Count);
constexpr size_t PACKET_KINDS = 256; // PM4 type-3 opcodes.

// GameFlex: who asks Image::Transit for a layout and an access. A call that records no barrier is booked under its
// caller's cause (targets, bindings), so its own cost shows nowhere.
enum class TransitCaller : uint8_t {
	Targets,  // Render targets and depth.
	Bindings, // The images a draw or a dispatch binds.
	Heap,     // The resident image heap returning an image to the read state.
	Other,    // Copies, uploads, blits, resolves, presentation.
	Count
};

constexpr size_t TRANSIT_CALLER_COUNT = static_cast<size_t>(TransitCaller::Count);
// Image::Transit calls by whether they record a barrier, by path (the whole image at once, or subresource by
// subresource) and by caller.
constexpr size_t TRANSIT_KINDS = 2 * 2 * TRANSIT_CALLER_COUNT;

[[nodiscard]] constexpr size_t TransitIndex(bool barrier, bool per_subresource, TransitCaller caller) noexcept {
	return (static_cast<size_t>(barrier) * 2 + static_cast<size_t>(per_subresource)) * TRANSIT_CALLER_COUNT +
	       static_cast<size_t>(caller);
}

// Steps too frequent and too short for the cause clock: every call counted, about one in 16 timed.
enum class Timed : uint8_t {
	TransitFull,         // An Image::Transit of the whole image that records no barrier.
	TransitSubresources, // An Image::Transit subresource by subresource that records no barrier.
	DrawState,           // A draw's DrawRenderState built (value-initialized).
	Count
};

constexpr size_t TIMED_COUNT = static_cast<size_t>(Timed::Count);

// Where the last writer of the words the CPU waits for stands when the wait starts.
enum class Writer : uint8_t {
	Open,      // The command buffer being recorded: the wait submits it.
	Submitted, // Submitted, not executed yet.
	Executed,  // Executed, as the progress word says; the GPU has not signalled it done yet.
	Count
};

constexpr size_t WRITER_COUNT = static_cast<size_t>(Writer::Count);
// Waits for words the GPU writes, by the cause their time is booked under: WaitGpuSrt (a resource table), WaitGpuRead
// (any other read), or another one when an outer reason wins (a remap, a flip: WaitReasonScope).
constexpr size_t WORD_WAIT_CAUSES = 3;
constexpr size_t WORD_WAIT_COUNT  = WORD_WAIT_CAUSES * WRITER_COUNT;

// GameFlex (0094): a submission's kind, as RADV sees its command buffer. Binding a pipeline with a task stage (an
// indirect mesh draw) gives the buffer an ACE stream that a reset keeps: from then on, every submission of that
// VkCommandBuffer goes to the kernel as a gang, and rebuilds the ACE queue's preambles (chantier 48, P4.5).
enum class SubmitKind : uint8_t {
	Task,     // It bound a mesh_task pipeline in this recording.
	Marked,   // Its VkCommandBuffer bound one in an earlier recording, not in this one.
	Ordinary, // Neither.
	Count
};

constexpr size_t SUBMIT_KIND_COUNT = static_cast<size_t>(SubmitKind::Count);

// The size of a submitted buffer, in draws and dispatches: for a cost of the form a + b x size.
enum class SubmitSize : uint8_t {
	Small,  // Under 64.
	Medium, // 64 to 512.
	Large,  // Over 512.
	Count
};

constexpr size_t SUBMIT_SIZE_COUNT = static_cast<size_t>(SubmitSize::Count);
constexpr size_t SUBMIT_CELLS      = SUBMIT_KIND_COUNT * SUBMIT_SIZE_COUNT;

[[nodiscard]] constexpr SubmitSize SizeOf(uint64_t commands) noexcept {
	return commands < 64 ? SubmitSize::Small : commands <= 512 ? SubmitSize::Medium : SubmitSize::Large;
}

[[nodiscard]] constexpr size_t SubmitCell(SubmitKind kind, SubmitSize size) noexcept {
	return static_cast<size_t>(kind) * SUBMIT_SIZE_COUNT + static_cast<size_t>(size);
}

// GameFlex (0094): where the GPU stood around a buffer's vkQueueSubmit, for the gap before the GPU starts it.
enum class GpuGap : uint8_t {
	DryBefore, // Every earlier buffer was done before the call.
	DryDuring, // Not before the call, but by its return (the progress word read again).
	Queued,    // An earlier buffer still ran when the call returned.
	Count
};

constexpr size_t GPU_GAP_COUNT = static_cast<size_t>(GpuGap::Count);
// By where the GPU stood, then by the kind of the buffer after the gap.
constexpr size_t GPU_GAP_CELLS = GPU_GAP_COUNT * SUBMIT_KIND_COUNT;

[[nodiscard]] constexpr size_t GapCell(GpuGap gap, SubmitKind kind) noexcept {
	return static_cast<size_t>(gap) * SUBMIT_KIND_COUNT + static_cast<size_t>(kind);
}

// GameFlex (0094): the buffers on the GPU, submitted and not executed, when one closes: 0, 1, 2 or more.
constexpr size_t   QUEUED_DEPTHS = 3;
constexpr uint64_t DEPTH_UNKNOWN = ~uint64_t {0}; // No progress word: asking the driver would cost a system call.
// A game thread's take of GuestGpu's queue lock that lasted this long waited for it (about 1 us of the TSC).
constexpr uint64_t GUEST_LOCK_WAIT_TICKS = 4096;

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
// GameFlex (0088): what the P4 pieces are decided on.
inline std::atomic<uint64_t> g_empty_closes[CLOSE_COUNT] {}; // Buffers submitted with nothing recorded in them.
inline std::atomic<uint64_t> g_empty_operation_closes[CLOSE_COUNT] {}; // Of those, the ones an operation waits for.
inline std::atomic<uint64_t> g_timestamp_ticks {0};          // The GPU timestamps, inside submit begin and end.
inline std::atomic<uint64_t> g_transits[TRANSIT_KINDS] {};
inline std::atomic<uint64_t> g_timed_calls[TIMED_COUNT] {};
inline std::atomic<uint64_t> g_timed_samples[TIMED_COUNT] {}; // The calls timed.
inline std::atomic<uint64_t> g_timed_ticks[TIMED_COUNT] {};   // Their time, the two clock reads included.
inline std::atomic<uint64_t> g_clock_pairs {0};               // Back-to-back pairs of clock reads (CalibrateClock).
inline std::atomic<uint64_t> g_clock_pair_ticks {0};          // Their time.
inline std::atomic<uint64_t> g_word_waits[WORD_WAIT_COUNT] {};
inline std::atomic<uint64_t> g_word_wait_ticks[WORD_WAIT_COUNT] {};     // The wait itself: its cause's share.
inline std::atomic<uint64_t> g_word_finish_ticks[WORD_WAIT_COUNT] {};   // The whole Finish: submit, begin, operations.
inline std::atomic<uint64_t> g_word_wait_ahead[WORD_WAIT_COUNT] {};     // Buffers the GPU executes up to the writer.
inline std::atomic<uint64_t> g_word_wait_behind[WORD_WAIT_COUNT] {};    // Buffers after it, the open one included.
inline std::atomic<uint64_t> g_word_wait_open_used[WORD_WAIT_COUNT] {}; // Waits with work in the open buffer.
// The stream's waits for a flip done (WaitFlipDone): without, then with a flip of that buffer pending before the
// flush that precedes them.
inline std::atomic<uint64_t> g_flip_waits[2] {};
inline std::atomic<uint64_t> g_flip_wait_ticks[2] {}; // Their share of WaitGpuFrame.
// The game's suspend points: taken by this thread, released once the GPU has done everything before them.
inline std::atomic<uint64_t> g_suspend_taken {0};
inline std::atomic<uint64_t> g_suspend_taken_ns {0};      // From the game's call.
inline std::atomic<uint64_t> g_suspend_taken_buffers {0}; // Buffers submitted and not executed, when taken.
inline std::atomic<uint64_t> g_suspend_released {0};
inline std::atomic<uint64_t> g_suspend_released_ns {0}; // From being taken.
inline std::atomic<uint64_t> g_suspend_released_buffers {0};
// GameFlex (0094): the submissions of P4.5, by kind and size (SubmitCell): vkQueueSubmit alone, this thread's CPU
// time and context switches during it, and what the buffer held.
inline std::atomic<uint64_t> g_submits[SUBMIT_CELLS] {};
inline std::atomic<uint64_t> g_submit_call_ticks[SUBMIT_CELLS] {};
inline std::atomic<uint64_t> g_submit_cpu_ns[SUBMIT_CELLS] {};
inline std::atomic<uint64_t> g_submit_voluntary[SUBMIT_CELLS] {};
inline std::atomic<uint64_t> g_submit_involuntary[SUBMIT_CELLS] {};
inline std::atomic<uint64_t> g_submit_draws[SUBMIT_CELLS] {};
inline std::atomic<uint64_t> g_submit_dispatches[SUBMIT_CELLS] {};
inline std::atomic<uint64_t> g_submit_packets[SUBMIT_CELLS] {};
inline std::atomic<uint64_t> g_submit_pipelines[SUBMIT_CELLS] {}; // Distinct pipelines its draws and dispatches bound.
// By close reason: the buffers on the GPU when it closed (QUEUED_DEPTHS each), and vkQueueSubmit alone.
inline std::atomic<uint64_t> g_close_depths[CLOSE_COUNT * QUEUED_DEPTHS] {};
inline std::atomic<uint64_t> g_close_samples[CLOSE_COUNT] {};
inline std::atomic<uint64_t> g_close_call_ticks[CLOSE_COUNT] {};
// From the GPU timestamps: busy time by kind; the gaps by GapCell (every buffer measured after the one before it
// counts, gap or not); the delay from the call's return to the buffer's start on the GPU (calibrated timestamps),
// signed, summed modulo 2^64.
inline std::atomic<uint64_t> g_gpu_kind_busy_ns[SUBMIT_KIND_COUNT] {};
inline std::atomic<uint64_t> g_gpu_kind_buffers[SUBMIT_KIND_COUNT] {};
inline std::atomic<uint64_t> g_gpu_gap_cell_ns[GPU_GAP_CELLS] {};
inline std::atomic<uint64_t> g_gpu_gap_cell_buffers[GPU_GAP_CELLS] {};
inline std::atomic<uint64_t> g_start_delay_ns[GPU_GAP_CELLS] {};
inline std::atomic<uint64_t> g_start_delays[GPU_GAP_CELLS] {};
inline std::atomic<bool>     g_calibrated {false};
inline std::atomic<uint64_t> g_calibration_deviation_ns {0}; // The last calibration's.
// mesh_task pipelines bound; the first one, in ns after the command thread began, plus 1 (0: none yet); the command
// pool's buffers and those a task stage marked.
inline std::atomic<uint64_t> g_task_binds {0};
inline std::atomic<uint64_t> g_thread_start_ns {0};
inline std::atomic<uint64_t> g_first_task_ns {0};
inline std::atomic<uint64_t> g_pool_buffers {0};
inline std::atomic<uint64_t> g_pool_marked {0};
// The game's threads on GuestGpu's queue lock (SendCommand, Enqueue): every take, and those that waited.
inline std::atomic<uint64_t> g_guest_locks {0};
inline std::atomic<uint64_t> g_guest_lock_ticks {0};
inline std::atomic<uint64_t> g_guest_lock_waits {0};
inline std::atomic<uint64_t> g_guest_lock_wait_ticks {0};

constexpr uint32_t CLOCK_PAIRS             = 1024; // When the thread starts.
constexpr uint32_t CLOCK_PAIRS_RECALIBRATE = 64;   // At each suspend point.

struct ThreadState {
	bool     command_thread = false;
	bool     counting       = false; // The overlay was on at the last transition.
	Cause      current        = Cause::Other;
	WaitReason reason         = WaitReason::Other;
	bool       label_gpu      = true; // What the last suspended guest wait waits for.
	uint64_t   since          = 0;
	uint32_t   sample         = 1; // The pseudo-random sequence that picks the calls timed (TimedStart).
};
inline thread_local ThreadState t_state;

[[nodiscard]] inline uint64_t Ticks() noexcept {
#if defined(_M_X64) || defined(__x86_64__)
	return __rdtsc();
#else
	return SteadyNs();
#endif
}

// GameFlex: the clock for steps of a few nanoseconds. rdtsc alone runs ahead of or behind the step's own instructions,
// which then vanish in its latency: the fences keep the step between the two reads.
[[nodiscard]] inline uint64_t OrderedTicks() noexcept {
#if defined(_M_X64) || defined(__x86_64__)
	_mm_lfence();
	const auto ticks = __rdtsc();
	_mm_lfence();
	return ticks;
#else
	return SteadyNs();
#endif
}

// One writer, this thread: a plain add, readers see a recent value.
inline void AddOwned(std::atomic<uint64_t>& counter, uint64_t value) noexcept {
	counter.store(counter.load(std::memory_order_relaxed) + value, std::memory_order_relaxed);
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
			AddOwned(g_ticks[static_cast<size_t>(state.current)], now - state.since);
		}
		state.since = now;
	}
	state.counting = counting;
	state.current  = next;
}

// A call of a frequent step counted, with its time when it was one of those timed (start != 0).
inline void AddTimed(Timed step, uint64_t start, uint64_t end) noexcept {
	const auto index = static_cast<size_t>(step);
	AddOwned(g_timed_calls[index], 1);
	if (start != 0) {
		AddOwned(g_timed_ticks[index], end - start);
		AddOwned(g_timed_samples[index], 1);
	}
}

// The clock at the end of a step timed from `start`, or 0 when it was not timed.
[[nodiscard]] inline uint64_t TimedStop(uint64_t start) noexcept {
	if (start == 0) {
		return 0;
	}
	std::atomic_signal_fence(std::memory_order_seq_cst);
	return OrderedTicks();
}

// GameFlex: what the timed steps' own two clock reads cost, taken out of their times in the report. The TSC runs at a
// fixed rate while lfence and rdtsc take a fixed number of core cycles: the cost follows the core's clock, so it is
// measured again while the game runs and the report takes the pairs of its own window.
inline void CalibrateClock(uint32_t pairs) noexcept {
	uint64_t ticks = 0;
	for (uint32_t i = 0; i < pairs; i++) {
		const auto start = OrderedTicks();
		std::atomic_signal_fence(std::memory_order_seq_cst);
		ticks += OrderedTicks() - start;
	}
	AddOwned(g_clock_pair_ticks, ticks);
	AddOwned(g_clock_pairs, pairs);
}
} // namespace Detail

// Called once by the command thread, before its first command.
inline void SetCommandThread() noexcept {
	Detail::t_state.command_thread = true;
	Detail::t_state.current        = Cause::Other;
	Detail::g_thread_start_ns.store(SteadyNs(), std::memory_order_relaxed);
	// GameFlex: the clock reads' cost for a report whose window holds no suspend point (RecalibrateClock).
	Detail::CalibrateClock(Detail::CLOCK_PAIRS);
}

[[nodiscard]] inline bool OnCommandThread() noexcept {
	return Detail::t_state.command_thread;
}

// The command thread counts now: for callers that gather what they count at some cost.
[[nodiscard]] inline bool Counting() noexcept {
	return Detail::t_state.command_thread && DebugCounters::Enabled();
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

// A command buffer submitted, and why; `empty` when nothing was recorded in it, `operations` when a deferred
// operation waits for it all the same (an empty buffer can stay open only when none does).
inline void CountClose(Close reason, bool empty, bool operations) noexcept {
	if (DebugCounters::Enabled()) {
		Detail::g_closes[static_cast<size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
		if (empty) {
			Detail::g_empty_closes[static_cast<size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
			if (operations) {
				Detail::g_empty_operation_closes[static_cast<size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
			}
		}
	}
}

// GameFlex: the clock, for the instrument's own steps timed each time (they are few).
[[nodiscard]] inline uint64_t Clock() noexcept {
	return Detail::Ticks();
}

// GameFlex: the time this thread booked under `cause` so far. Around a call, the difference is the call's share of
// the cause: a wait's own time, without the work around it. One writer, this thread: on the command thread only.
[[nodiscard]] inline uint64_t CauseTicks(Cause cause) noexcept {
	return Detail::g_ticks[static_cast<size_t>(cause)].load(std::memory_order_relaxed);
}

// The instrument's own GPU timestamps (CommandScheduler), inside submit begin and end: their cost apart.
inline void AddTimestampTicks(uint64_t start) noexcept {
	Detail::g_timestamp_ticks.fetch_add(Detail::Ticks() - start, std::memory_order_relaxed);
}

// The clock at the start of a call of a frequent step on the command thread when this call is one of those timed,
// else 0. Two clock reads cost about what such a step does, so about one call in 16 is timed, picked at random: a
// fixed stride would time the same call of every draw. The others are only counted.
[[nodiscard]] inline uint64_t TimedStart() noexcept {
	auto& state = Detail::t_state;
	if (!state.command_thread || !DebugCounters::Enabled()) {
		return 0;
	}
	state.sample = state.sample * 1664525u + 1013904223u;
	if ((state.sample >> 28u) != 0) {
		return 0;
	}
	const auto start = Detail::OrderedTicks();
	// The step's own memory accesses stay between the two clock reads.
	std::atomic_signal_fence(std::memory_order_seq_cst);
	return start;
}

// The end of a call of a frequent step that TimedStart began.
inline void TimedEnd(Timed step, uint64_t start) noexcept {
	const auto end = Detail::TimedStop(start);
	if (Counting()) {
		Detail::AddTimed(step, start, end);
	}
}

// GameFlex: the clock reads' cost measured again on the command thread, at the clock the core runs at in the game
// (CalibrateClock). A few microseconds, at each suspend point.
inline void RecalibrateClock() noexcept {
	if (Counting()) {
		Detail::CalibrateClock(Detail::CLOCK_PAIRS_RECALIBRATE);
	}
}

// An Image::Transit call that TimedStart began: whether it recorded a barrier, its path and its caller, and the time
// of one that recorded none.
inline void CountTransit(TransitCaller caller, bool per_subresource, bool barrier, uint64_t start) noexcept {
	const auto end = Detail::TimedStop(start);
	if (!Counting()) {
		return;
	}
	Detail::AddOwned(Detail::g_transits[TransitIndex(barrier, per_subresource, caller)], 1);
	if (!barrier) {
		Detail::AddTimed(per_subresource ? Timed::TransitSubresources : Timed::TransitFull, start, end);
	}
}

// A wait of the command thread for words the GPU writes, from StartWordWait to EndWordWait.
struct WordWait {
	size_t   index       = WORD_WAIT_COUNT; // WORD_WAIT_COUNT: not counted.
	Cause    cause       = Cause::Other;    // The cause its time is booked under.
	uint64_t cause_ticks = 0;               // That cause's time when it started.
	uint64_t start       = 0;
};

// Only while Counting(), under the WaitReasonScope of the wait. `writer_tick` is the buffer that writes the words
// last, `current_tick` the open buffer, `done_tick` the last one the GPU executed, read before `current_tick`;
// `open_used` says whether anything was recorded in the open buffer. The wait (Finish) submits the open buffer and
// waits for every buffer up to it: those up to the writer, which a wait for the writer alone would need, and those
// after it, the open one included.
[[nodiscard]] inline WordWait StartWordWait(uint64_t writer_tick, uint64_t current_tick, uint64_t done_tick,
                                            bool open_used) noexcept {
	const auto cause  = WaitCause();
	const auto group  = static_cast<size_t>(cause == Cause::WaitGpuSrt ? 0 : cause == Cause::WaitGpuRead ? 1 : 2);
	done_tick         = std::min(done_tick, current_tick - 1);
	const auto writer = writer_tick >= current_tick ? Writer::Open
	                    : writer_tick > done_tick   ? Writer::Submitted
	                                                : Writer::Executed;
	const auto index  = group * WRITER_COUNT + static_cast<size_t>(writer);
	uint64_t   ahead  = current_tick - done_tick;
	uint64_t   behind = 0;
	if (writer != Writer::Open) {
		const auto reached = std::max(writer_tick, done_tick);
		ahead              = reached - done_tick;
		behind             = current_tick - reached;
	}
	Detail::AddOwned(Detail::g_word_waits[index], 1);
	Detail::AddOwned(Detail::g_word_wait_ahead[index], ahead);
	Detail::AddOwned(Detail::g_word_wait_behind[index], behind);
	if (open_used) {
		Detail::AddOwned(Detail::g_word_wait_open_used[index], 1);
	}
	return {index, cause, CauseTicks(cause), Detail::Ticks()};
}

// The wait's time is what its cause gained meanwhile: the semaphore wait alone. The whole Finish (the open buffer
// submitted, the next one begun, the deferred operations run) is counted apart.
inline void EndWordWait(const WordWait& wait) noexcept {
	if (wait.index < WORD_WAIT_COUNT) {
		Detail::AddOwned(Detail::g_word_finish_ticks[wait.index], Detail::Ticks() - wait.start);
		Detail::AddOwned(Detail::g_word_wait_ticks[wait.index], CauseTicks(wait.cause) - wait.cause_ticks);
	}
}

// GameFlex: the stream waited for a flip done (WaitFlipDone), on the command thread: `pending` when a flip of that
// buffer was pending before the flush that precedes the wait (without one, the flush gave the GPU nothing the wait
// needs), `ticks` the wait's share of WaitGpuFrame.
inline void NoteFlipWait(bool pending, uint64_t ticks) noexcept {
	Detail::AddOwned(Detail::g_flip_waits[pending ? 1 : 0], 1);
	Detail::AddOwned(Detail::g_flip_wait_ticks[pending ? 1 : 0], ticks);
}

// The game's suspend points cross threads (the game's, this one, the one that releases them): their clock is in
// nanoseconds, and 0 while nothing counts.
[[nodiscard]] inline uint64_t SuspendClock() noexcept {
	return DebugCounters::Enabled() ? SteadyNs() : 0;
}

// The command thread took at `taken_ns` a suspend point the game called at `called_ns`, `queued` buffers submitted
// and not executed.
inline void NoteSuspendPointTaken(uint64_t called_ns, uint64_t taken_ns, uint64_t queued) noexcept {
	Detail::g_suspend_taken.fetch_add(1, std::memory_order_relaxed);
	Detail::g_suspend_taken_ns.fetch_add(called_ns != 0 && taken_ns > called_ns ? taken_ns - called_ns : 0,
	                                     std::memory_order_relaxed);
	Detail::g_suspend_taken_buffers.fetch_add(queued, std::memory_order_relaxed);
}

// The point taken at `taken_ns` is released: the GPU executed everything before it, `queued` buffers still on it.
inline void NoteSuspendPointReleased(uint64_t taken_ns, uint64_t queued) noexcept {
	const auto now = SteadyNs();
	Detail::g_suspend_released.fetch_add(1, std::memory_order_relaxed);
	Detail::g_suspend_released_ns.fetch_add(now > taken_ns ? now - taken_ns : 0, std::memory_order_relaxed);
	Detail::g_suspend_released_buffers.fetch_add(queued, std::memory_order_relaxed);
}

// GameFlex (0094): an event counter's total so far, for the share of one command buffer.
[[nodiscard]] inline uint64_t EventTotal(Event event) noexcept {
	return Detail::g_events[static_cast<size_t>(event)].load(std::memory_order_relaxed);
}

// GameFlex (0094): one submission of the command thread, only while Counting().
struct SubmitSample {
	SubmitKind kind        = SubmitKind::Ordinary;
	Close      reason      = Close::Other;
	uint64_t   depth       = DEPTH_UNKNOWN; // Buffers on the GPU when it closed.
	uint64_t   call_ticks  = 0;             // vkQueueSubmit alone.
	uint64_t   cpu_ns      = 0;             // This thread's CPU time during the call.
	uint64_t   voluntary   = 0;             // Its context switches during the call.
	uint64_t   involuntary = 0;
	uint64_t   draws       = 0;
	uint64_t   dispatches  = 0;
	uint64_t   packets     = 0;
	uint64_t   pipelines   = 0;
};

inline void NoteSubmit(const SubmitSample& sample) noexcept {
	using namespace Detail;
	const auto cell = SubmitCell(sample.kind, SizeOf(sample.draws + sample.dispatches));
	AddOwned(g_submits[cell], 1);
	AddOwned(g_submit_call_ticks[cell], sample.call_ticks);
	AddOwned(g_submit_cpu_ns[cell], sample.cpu_ns);
	AddOwned(g_submit_voluntary[cell], sample.voluntary);
	AddOwned(g_submit_involuntary[cell], sample.involuntary);
	AddOwned(g_submit_draws[cell], sample.draws);
	AddOwned(g_submit_dispatches[cell], sample.dispatches);
	AddOwned(g_submit_packets[cell], sample.packets);
	AddOwned(g_submit_pipelines[cell], sample.pipelines);
	const auto reason = static_cast<size_t>(sample.reason);
	AddOwned(g_close_samples[reason], 1);
	AddOwned(g_close_call_ticks[reason], sample.call_ticks);
	if (sample.depth != DEPTH_UNKNOWN) {
		AddOwned(g_close_depths[reason * QUEUED_DEPTHS + std::min<uint64_t>(sample.depth, QUEUED_DEPTHS - 1)], 1);
	}
}

// GameFlex (0094): what HarvestTimestamps found by kind and by gap (GapCell), in ns; `delay_ns` signed, modulo 2^64.
struct GpuCells {
	std::array<uint64_t, SUBMIT_KIND_COUNT> busy_ns {};
	std::array<uint64_t, SUBMIT_KIND_COUNT> buffers {};
	std::array<uint64_t, GPU_GAP_CELLS>     gap_ns {};
	std::array<uint64_t, GPU_GAP_CELLS>     gap_buffers {};
	std::array<uint64_t, GPU_GAP_CELLS>     delay_ns {};
	std::array<uint64_t, GPU_GAP_CELLS>     delays {};
};

inline void AddGpuCells(const GpuCells& cells) noexcept {
	using namespace Detail;
	for (size_t i = 0; i < SUBMIT_KIND_COUNT; i++) {
		g_gpu_kind_busy_ns[i].fetch_add(cells.busy_ns[i], std::memory_order_relaxed);
		g_gpu_kind_buffers[i].fetch_add(cells.buffers[i], std::memory_order_relaxed);
	}
	for (size_t i = 0; i < GPU_GAP_CELLS; i++) {
		g_gpu_gap_cell_ns[i].fetch_add(cells.gap_ns[i], std::memory_order_relaxed);
		g_gpu_gap_cell_buffers[i].fetch_add(cells.gap_buffers[i], std::memory_order_relaxed);
		g_start_delay_ns[i].fetch_add(cells.delay_ns[i], std::memory_order_relaxed);
		g_start_delays[i].fetch_add(cells.delays[i], std::memory_order_relaxed);
	}
}

// GameFlex (0094): the GPU's timestamps can be related to the CPU's clock (VK_KHR_calibrated_timestamps); the last
// calibration's maximum deviation.
inline void NoteCalibration(uint64_t deviation_ns) noexcept {
	Detail::g_calibrated.store(true, std::memory_order_relaxed);
	Detail::g_calibration_deviation_ns.store(deviation_ns, std::memory_order_relaxed);
}

// GameFlex (0094): the command thread bound a mesh_task pipeline; `marked` buffers of the pool's `buffers` have.
// Counted whether the overlay counts or not: the first one's time and the marks are the game's, not the window's.
inline void NoteTaskBind(uint64_t marked, uint64_t buffers) noexcept {
	using namespace Detail;
	if (g_first_task_ns.load(std::memory_order_relaxed) == 0) {
		g_first_task_ns.store(SteadyNs() - g_thread_start_ns.load(std::memory_order_relaxed) + 1,
		                      std::memory_order_relaxed);
	}
	g_pool_marked.store(marked, std::memory_order_relaxed);
	g_pool_buffers.store(buffers, std::memory_order_relaxed);
	if (DebugCounters::Enabled()) {
		g_task_binds.fetch_add(1, std::memory_order_relaxed);
	}
}

// GameFlex (0094): the command pool's size, as it grows.
inline void NotePoolBuffers(uint64_t buffers) noexcept {
	Detail::g_pool_buffers.store(buffers, std::memory_order_relaxed);
}

// GameFlex (0094): a game thread takes GuestGpu's queue lock (SendCommand, Enqueue), which the command thread holds
// while it submits before sleeping (FlushBeforeIdle). The clock before the lock; 0 while nothing counts, and on the
// command thread itself.
[[nodiscard]] inline uint64_t GuestLockStart() noexcept {
	return DebugCounters::Enabled() && !Detail::t_state.command_thread ? Detail::Ticks() : 0;
}

inline void GuestLockTaken(uint64_t start) noexcept {
	if (start == 0) {
		return;
	}
	using namespace Detail;
	const auto ticks = Ticks() - start;
	g_guest_locks.fetch_add(1, std::memory_order_relaxed);
	g_guest_lock_ticks.fetch_add(ticks, std::memory_order_relaxed);
	if (ticks >= GUEST_LOCK_WAIT_TICKS) {
		g_guest_lock_waits.fetch_add(1, std::memory_order_relaxed);
		g_guest_lock_wait_ticks.fetch_add(ticks, std::memory_order_relaxed);
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
	// GameFlex (0088).
	std::array<uint64_t, CLOSE_COUNT>     empty_closes {};
	std::array<uint64_t, CLOSE_COUNT>     empty_operation_closes {};
	uint64_t                              timestamp_ticks = 0;
	std::array<uint64_t, TRANSIT_KINDS>   transits {};
	std::array<uint64_t, TIMED_COUNT>     timed_calls {};
	std::array<uint64_t, TIMED_COUNT>     timed_samples {};
	std::array<uint64_t, TIMED_COUNT>     timed_ticks {};
	uint64_t                              clock_pairs      = 0;
	uint64_t                              clock_pair_ticks = 0;
	std::array<uint64_t, WORD_WAIT_COUNT> word_waits {};
	std::array<uint64_t, WORD_WAIT_COUNT> word_wait_ticks {};
	std::array<uint64_t, WORD_WAIT_COUNT> word_finish_ticks {};
	std::array<uint64_t, WORD_WAIT_COUNT> word_wait_ahead {};
	std::array<uint64_t, WORD_WAIT_COUNT> word_wait_behind {};
	std::array<uint64_t, WORD_WAIT_COUNT> word_wait_open_used {};
	std::array<uint64_t, 2>               flip_waits {};
	std::array<uint64_t, 2>               flip_wait_ticks {};
	uint64_t                              suspend_taken            = 0;
	uint64_t                              suspend_taken_ns         = 0;
	uint64_t                              suspend_taken_buffers    = 0;
	uint64_t                              suspend_released         = 0;
	uint64_t                              suspend_released_ns      = 0;
	uint64_t                              suspend_released_buffers = 0;
	// GameFlex (0094).
	std::array<uint64_t, SUBMIT_CELLS>                  submits {};
	std::array<uint64_t, SUBMIT_CELLS>                  submit_call_ticks {};
	std::array<uint64_t, SUBMIT_CELLS>                  submit_cpu_ns {};
	std::array<uint64_t, SUBMIT_CELLS>                  submit_voluntary {};
	std::array<uint64_t, SUBMIT_CELLS>                  submit_involuntary {};
	std::array<uint64_t, SUBMIT_CELLS>                  submit_draws {};
	std::array<uint64_t, SUBMIT_CELLS>                  submit_dispatches {};
	std::array<uint64_t, SUBMIT_CELLS>                  submit_packets {};
	std::array<uint64_t, SUBMIT_CELLS>                  submit_pipelines {};
	std::array<uint64_t, CLOSE_COUNT * QUEUED_DEPTHS>   close_depths {};
	std::array<uint64_t, CLOSE_COUNT>                   close_samples {};
	std::array<uint64_t, CLOSE_COUNT>                   close_call_ticks {};
	std::array<uint64_t, SUBMIT_KIND_COUNT>             gpu_kind_busy_ns {};
	std::array<uint64_t, SUBMIT_KIND_COUNT>             gpu_kind_buffers {};
	std::array<uint64_t, GPU_GAP_CELLS>                 gpu_gap_cell_ns {};
	std::array<uint64_t, GPU_GAP_CELLS>                 gpu_gap_cell_buffers {};
	std::array<uint64_t, GPU_GAP_CELLS>                 start_delay_ns {};
	std::array<uint64_t, GPU_GAP_CELLS>                 start_delays {};
	bool                                                calibrated               = false;
	uint64_t                                            calibration_deviation_ns = 0;
	uint64_t                                            task_binds               = 0;
	uint64_t                                            thread_start_ns          = 0;
	uint64_t                                            first_task_ns            = 0;
	uint64_t                                            pool_buffers             = 0;
	uint64_t                                            pool_marked              = 0;
	uint64_t                                            guest_locks              = 0;
	uint64_t                                            guest_lock_ticks         = 0;
	uint64_t                                            guest_lock_waits         = 0;
	uint64_t                                            guest_lock_wait_ticks    = 0;
};

namespace Detail {
template <size_t N>
void LoadAll(const std::atomic<uint64_t> (&counters)[N], std::array<uint64_t, N>& values) noexcept {
	for (size_t i = 0; i < N; i++) {
		values[i] = counters[i].load(std::memory_order_relaxed);
	}
}
} // namespace Detail

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
	Detail::LoadAll(Detail::g_empty_closes, totals.empty_closes);
	Detail::LoadAll(Detail::g_empty_operation_closes, totals.empty_operation_closes);
	totals.timestamp_ticks = Detail::g_timestamp_ticks.load(std::memory_order_relaxed);
	Detail::LoadAll(Detail::g_transits, totals.transits);
	Detail::LoadAll(Detail::g_timed_calls, totals.timed_calls);
	Detail::LoadAll(Detail::g_timed_samples, totals.timed_samples);
	Detail::LoadAll(Detail::g_timed_ticks, totals.timed_ticks);
	totals.clock_pairs      = Detail::g_clock_pairs.load(std::memory_order_relaxed);
	totals.clock_pair_ticks = Detail::g_clock_pair_ticks.load(std::memory_order_relaxed);
	Detail::LoadAll(Detail::g_word_waits, totals.word_waits);
	Detail::LoadAll(Detail::g_word_wait_ticks, totals.word_wait_ticks);
	Detail::LoadAll(Detail::g_word_finish_ticks, totals.word_finish_ticks);
	Detail::LoadAll(Detail::g_word_wait_ahead, totals.word_wait_ahead);
	Detail::LoadAll(Detail::g_word_wait_behind, totals.word_wait_behind);
	Detail::LoadAll(Detail::g_word_wait_open_used, totals.word_wait_open_used);
	Detail::LoadAll(Detail::g_flip_waits, totals.flip_waits);
	Detail::LoadAll(Detail::g_flip_wait_ticks, totals.flip_wait_ticks);
	totals.suspend_taken            = Detail::g_suspend_taken.load(std::memory_order_relaxed);
	totals.suspend_taken_ns         = Detail::g_suspend_taken_ns.load(std::memory_order_relaxed);
	totals.suspend_taken_buffers    = Detail::g_suspend_taken_buffers.load(std::memory_order_relaxed);
	totals.suspend_released         = Detail::g_suspend_released.load(std::memory_order_relaxed);
	totals.suspend_released_ns      = Detail::g_suspend_released_ns.load(std::memory_order_relaxed);
	totals.suspend_released_buffers = Detail::g_suspend_released_buffers.load(std::memory_order_relaxed);
	// GameFlex (0094).
	Detail::LoadAll(Detail::g_submits, totals.submits);
	Detail::LoadAll(Detail::g_submit_call_ticks, totals.submit_call_ticks);
	Detail::LoadAll(Detail::g_submit_cpu_ns, totals.submit_cpu_ns);
	Detail::LoadAll(Detail::g_submit_voluntary, totals.submit_voluntary);
	Detail::LoadAll(Detail::g_submit_involuntary, totals.submit_involuntary);
	Detail::LoadAll(Detail::g_submit_draws, totals.submit_draws);
	Detail::LoadAll(Detail::g_submit_dispatches, totals.submit_dispatches);
	Detail::LoadAll(Detail::g_submit_packets, totals.submit_packets);
	Detail::LoadAll(Detail::g_submit_pipelines, totals.submit_pipelines);
	Detail::LoadAll(Detail::g_close_depths, totals.close_depths);
	Detail::LoadAll(Detail::g_close_samples, totals.close_samples);
	Detail::LoadAll(Detail::g_close_call_ticks, totals.close_call_ticks);
	Detail::LoadAll(Detail::g_gpu_kind_busy_ns, totals.gpu_kind_busy_ns);
	Detail::LoadAll(Detail::g_gpu_kind_buffers, totals.gpu_kind_buffers);
	Detail::LoadAll(Detail::g_gpu_gap_cell_ns, totals.gpu_gap_cell_ns);
	Detail::LoadAll(Detail::g_gpu_gap_cell_buffers, totals.gpu_gap_cell_buffers);
	Detail::LoadAll(Detail::g_start_delay_ns, totals.start_delay_ns);
	Detail::LoadAll(Detail::g_start_delays, totals.start_delays);
	totals.calibrated               = Detail::g_calibrated.load(std::memory_order_relaxed);
	totals.calibration_deviation_ns = Detail::g_calibration_deviation_ns.load(std::memory_order_relaxed);
	totals.task_binds               = Detail::g_task_binds.load(std::memory_order_relaxed);
	totals.thread_start_ns          = Detail::g_thread_start_ns.load(std::memory_order_relaxed);
	totals.first_task_ns            = Detail::g_first_task_ns.load(std::memory_order_relaxed);
	totals.pool_buffers             = Detail::g_pool_buffers.load(std::memory_order_relaxed);
	totals.pool_marked              = Detail::g_pool_marked.load(std::memory_order_relaxed);
	totals.guest_locks              = Detail::g_guest_locks.load(std::memory_order_relaxed);
	totals.guest_lock_ticks         = Detail::g_guest_lock_ticks.load(std::memory_order_relaxed);
	totals.guest_lock_waits         = Detail::g_guest_lock_waits.load(std::memory_order_relaxed);
	totals.guest_lock_wait_ticks    = Detail::g_guest_lock_wait_ticks.load(std::memory_order_relaxed);
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
	// GameFlex (0088).
	std::array<double, CLOSE_COUNT>   empty_closes_per_frame {};
	std::array<double, CLOSE_COUNT>   empty_operation_closes_per_frame {};
	double                            timestamp_ms_per_frame = 0.0;
	std::array<double, TRANSIT_KINDS> transits_per_frame {};
	double                            clock_pair_ticks = 0.0; // Two clock reads, the window's own calibrations.
	std::array<double, TIMED_COUNT>   timed_per_frame {};
	std::array<double, TIMED_COUNT>   timed_samples_per_frame {};
	std::array<double, TIMED_COUNT>   timed_ticks_per_call {}; // Raw, the clock reads included.
	std::array<double, TIMED_COUNT>   timed_ns_per_call {};    // The clock reads' own cost taken out, at least 0.
	std::array<double, TIMED_COUNT>   timed_ms_per_frame {};   // Every call at the timed calls' mean.
	std::array<double, WORD_WAIT_COUNT> word_waits_per_frame {};
	std::array<double, WORD_WAIT_COUNT> word_wait_ms_per_frame {};
	std::array<double, WORD_WAIT_COUNT> word_finish_ms_per_frame {};
	std::array<double, WORD_WAIT_COUNT> word_wait_ahead {};  // Buffers, per wait.
	std::array<double, WORD_WAIT_COUNT> word_wait_behind {}; // Buffers, per wait.
	std::array<double, WORD_WAIT_COUNT> word_wait_open_used_per_frame {};
	std::array<double, 2>               flip_waits_per_frame {};
	std::array<double, 2>               flip_wait_ms_per_frame {};
	double suspend_taken_per_frame    = 0.0;
	double suspend_taken_ms           = 0.0; // Per point, from the game's call.
	double suspend_taken_buffers      = 0.0; // Per point.
	double suspend_released_per_frame = 0.0;
	double suspend_released_ms        = 0.0; // Per point, from being taken.
	double suspend_released_buffers   = 0.0; // Per point.
	// GameFlex (0094). By SubmitCell: per frame, then per call or per buffer.
	std::array<double, SUBMIT_CELLS> submits_per_frame {};
	std::array<double, SUBMIT_CELLS> submit_call_us {};
	std::array<double, SUBMIT_CELLS> submit_draws {};
	std::array<double, SUBMIT_CELLS> submit_dispatches {};
	std::array<double, SUBMIT_CELLS> submit_packets {};
	std::array<double, SUBMIT_CELLS> submit_pipelines {};
	// By kind, the sizes together: per frame, then per call.
	std::array<double, SUBMIT_KIND_COUNT> kind_per_frame {};
	std::array<double, SUBMIT_KIND_COUNT> kind_call_us {};
	std::array<double, SUBMIT_KIND_COUNT> kind_cpu_us {};
	std::array<double, SUBMIT_KIND_COUNT> kind_voluntary {};
	std::array<double, SUBMIT_KIND_COUNT> kind_involuntary {};
	double                                call_us = 0.0; // Every call sampled.
	// By close reason: closes per frame by the buffers on the GPU (QUEUED_DEPTHS), vkQueueSubmit per call.
	std::array<double, CLOSE_COUNT * QUEUED_DEPTHS> close_depths_per_frame {};
	std::array<double, CLOSE_COUNT>                 close_call_us {};
	// The GPU: busy per buffer by kind; the gaps per frame and per buffer by GapCell, then by gap.
	std::array<double, SUBMIT_KIND_COUNT> gpu_kind_buffers_per_frame {};
	std::array<double, SUBMIT_KIND_COUNT> gpu_kind_busy_us {};
	std::array<double, GPU_GAP_CELLS>     gap_ms_per_frame {};
	std::array<double, GPU_GAP_CELLS>     gap_buffers_per_frame {};
	std::array<double, GPU_GAP_COUNT>     gap_class_ms_per_frame {};
	std::array<double, GPU_GAP_COUNT>     gap_class_buffers_per_frame {};
	std::array<double, GPU_GAP_COUNT>     gap_us_per_buffer {};       // Over the buffers of its class.
	std::array<double, GPU_GAP_COUNT>     gap_us_per_undry_buffer {}; // Over the buffers not dry before the call.
	// The delay from the call's return to the buffer's start on the GPU, per buffer; per frame, the buffers measured.
	bool                              calibrated   = false;
	double                            deviation_us = 0.0;
	std::array<double, GPU_GAP_CELLS> start_delay_us {};
	std::array<double, GPU_GAP_CELLS> start_delays_per_frame {};
	// mesh_task pipelines.
	double task_binds_per_frame = 0.0;
	double first_task_s         = -1.0; // After the command thread began; -1: none yet.
	double window_end_s         = 0.0;  // The same clock.
	double pool_buffers         = 0.0;
	double pool_marked          = 0.0;
	// The game's threads on GuestGpu's queue lock.
	double guest_locks_per_frame        = 0.0;
	double guest_lock_ms_per_frame      = 0.0;
	double guest_lock_waits_per_frame   = 0.0;
	double guest_lock_wait_ms_per_frame = 0.0;
	double guest_lock_wait_us           = 0.0; // GUEST_LOCK_WAIT_TICKS.
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
	// GameFlex (0088).
	const auto delta = [](uint64_t from, uint64_t to) { return static_cast<double>(to - from); };
	const auto mean  = [](double sum, double count) { return count > 0.0 ? sum / count : 0.0; };
	for (size_t i = 0; i < CLOSE_COUNT; i++) {
		report.empty_closes_per_frame[i] = delta(before.empty_closes[i], after.empty_closes[i]) * per_frame;
		report.empty_operation_closes_per_frame[i] =
		    delta(before.empty_operation_closes[i], after.empty_operation_closes[i]) * per_frame;
	}
	report.timestamp_ms_per_frame = delta(before.timestamp_ticks, after.timestamp_ticks) * ms_per_tick * per_frame;
	for (size_t i = 0; i < TRANSIT_KINDS; i++) {
		report.transits_per_frame[i] = delta(before.transits[i], after.transits[i]) * per_frame;
	}
	// The window's own calibrations, or every one so far when it took none (no suspend point in it).
	const auto pairs = delta(before.clock_pairs, after.clock_pairs);
	report.clock_pair_ticks =
	    pairs > 0.0 ? delta(before.clock_pair_ticks, after.clock_pair_ticks) / pairs
	                : mean(static_cast<double>(after.clock_pair_ticks), static_cast<double>(after.clock_pairs));
	for (size_t i = 0; i < TIMED_COUNT; i++) {
		const auto calls   = delta(before.timed_calls[i], after.timed_calls[i]);
		const auto samples = delta(before.timed_samples[i], after.timed_samples[i]);
		const auto ticks   = mean(delta(before.timed_ticks[i], after.timed_ticks[i]), samples);
		report.timed_per_frame[i]         = calls * per_frame;
		report.timed_samples_per_frame[i] = samples * per_frame;
		report.timed_ticks_per_call[i]    = ticks;
		report.timed_ns_per_call[i] =
		    samples > 0.0 ? std::max(0.0, ticks - report.clock_pair_ticks) * ms_per_tick * 1e6 : 0.0;
		report.timed_ms_per_frame[i] = report.timed_ns_per_call[i] / 1e6 * report.timed_per_frame[i];
	}
	for (size_t i = 0; i < WORD_WAIT_COUNT; i++) {
		const auto waits               = delta(before.word_waits[i], after.word_waits[i]);
		report.word_waits_per_frame[i] = waits * per_frame;
		report.word_wait_ms_per_frame[i] =
		    delta(before.word_wait_ticks[i], after.word_wait_ticks[i]) * ms_per_tick * per_frame;
		report.word_finish_ms_per_frame[i] =
		    delta(before.word_finish_ticks[i], after.word_finish_ticks[i]) * ms_per_tick * per_frame;
		report.word_wait_ahead[i]  = mean(delta(before.word_wait_ahead[i], after.word_wait_ahead[i]), waits);
		report.word_wait_behind[i] = mean(delta(before.word_wait_behind[i], after.word_wait_behind[i]), waits);
		report.word_wait_open_used_per_frame[i] =
		    delta(before.word_wait_open_used[i], after.word_wait_open_used[i]) * per_frame;
	}
	for (size_t i = 0; i < 2; i++) {
		report.flip_waits_per_frame[i] = delta(before.flip_waits[i], after.flip_waits[i]) * per_frame;
		report.flip_wait_ms_per_frame[i] =
		    delta(before.flip_wait_ticks[i], after.flip_wait_ticks[i]) * ms_per_tick * per_frame;
	}
	const auto taken                  = delta(before.suspend_taken, after.suspend_taken);
	const auto released               = delta(before.suspend_released, after.suspend_released);
	report.suspend_taken_per_frame    = taken * per_frame;
	report.suspend_taken_ms           = mean(delta(before.suspend_taken_ns, after.suspend_taken_ns), taken) / 1e6;
	report.suspend_taken_buffers      = mean(delta(before.suspend_taken_buffers, after.suspend_taken_buffers), taken);
	report.suspend_released_per_frame = released * per_frame;
	report.suspend_released_ms = mean(delta(before.suspend_released_ns, after.suspend_released_ns), released) / 1e6;
	report.suspend_released_buffers =
	    mean(delta(before.suspend_released_buffers, after.suspend_released_buffers), released);
	// GameFlex (0094).
	const double us_per_tick = ms_per_tick * 1000.0;
	double       calls_all   = 0.0;
	double       ticks_all   = 0.0;
	for (size_t kind = 0; kind < SUBMIT_KIND_COUNT; kind++) {
		double calls = 0.0, ticks = 0.0, cpu = 0.0, voluntary = 0.0, involuntary = 0.0;
		for (size_t size = 0; size < SUBMIT_SIZE_COUNT; size++) {
			const auto i                  = kind * SUBMIT_SIZE_COUNT + size;
			const auto n                  = delta(before.submits[i], after.submits[i]);
			const auto t                  = delta(before.submit_call_ticks[i], after.submit_call_ticks[i]);
			report.submits_per_frame[i]   = n * per_frame;
			report.submit_call_us[i]      = mean(t, n) * us_per_tick;
			report.submit_draws[i]        = mean(delta(before.submit_draws[i], after.submit_draws[i]), n);
			report.submit_dispatches[i]   = mean(delta(before.submit_dispatches[i], after.submit_dispatches[i]), n);
			report.submit_packets[i]      = mean(delta(before.submit_packets[i], after.submit_packets[i]), n);
			report.submit_pipelines[i]    = mean(delta(before.submit_pipelines[i], after.submit_pipelines[i]), n);
			calls += n;
			ticks += t;
			cpu += delta(before.submit_cpu_ns[i], after.submit_cpu_ns[i]);
			voluntary += delta(before.submit_voluntary[i], after.submit_voluntary[i]);
			involuntary += delta(before.submit_involuntary[i], after.submit_involuntary[i]);
		}
		report.kind_per_frame[kind]   = calls * per_frame;
		report.kind_call_us[kind]     = mean(ticks, calls) * us_per_tick;
		report.kind_cpu_us[kind]      = mean(cpu, calls) / 1000.0;
		report.kind_voluntary[kind]   = mean(voluntary, calls);
		report.kind_involuntary[kind] = mean(involuntary, calls);
		calls_all += calls;
		ticks_all += ticks;
	}
	report.call_us = mean(ticks_all, calls_all) * us_per_tick;
	for (size_t i = 0; i < CLOSE_COUNT * QUEUED_DEPTHS; i++) {
		report.close_depths_per_frame[i] = delta(before.close_depths[i], after.close_depths[i]) * per_frame;
	}
	for (size_t i = 0; i < CLOSE_COUNT; i++) {
		report.close_call_us[i] = mean(delta(before.close_call_ticks[i], after.close_call_ticks[i]),
		                               delta(before.close_samples[i], after.close_samples[i])) *
		                          us_per_tick;
	}
	for (size_t i = 0; i < SUBMIT_KIND_COUNT; i++) {
		const auto buffers                   = delta(before.gpu_kind_buffers[i], after.gpu_kind_buffers[i]);
		report.gpu_kind_buffers_per_frame[i] = buffers * per_frame;
		report.gpu_kind_busy_us[i] =
		    mean(delta(before.gpu_kind_busy_ns[i], after.gpu_kind_busy_ns[i]), buffers) / 1000.0;
	}
	std::array<double, GPU_GAP_COUNT> gap_ns {};
	std::array<double, GPU_GAP_COUNT> gap_buffers {};
	for (size_t i = 0; i < GPU_GAP_CELLS; i++) {
		const auto gap     = i / SUBMIT_KIND_COUNT;
		const auto ns      = delta(before.gpu_gap_cell_ns[i], after.gpu_gap_cell_ns[i]);
		const auto buffers = delta(before.gpu_gap_cell_buffers[i], after.gpu_gap_cell_buffers[i]);
		report.gap_ms_per_frame[i]      = ns / 1e6 * per_frame;
		report.gap_buffers_per_frame[i] = buffers * per_frame;
		gap_ns[gap] += ns;
		gap_buffers[gap] += buffers;
		const auto delays                = delta(before.start_delays[i], after.start_delays[i]);
		const auto delay_ns              = static_cast<double>(static_cast<int64_t>(after.start_delay_ns[i] - before.start_delay_ns[i]));
		report.start_delays_per_frame[i] = delays * per_frame;
		report.start_delay_us[i]         = mean(delay_ns, delays) / 1000.0;
	}
	const auto undry = gap_buffers[static_cast<size_t>(GpuGap::DryDuring)] + gap_buffers[static_cast<size_t>(GpuGap::Queued)];
	for (size_t gap = 0; gap < GPU_GAP_COUNT; gap++) {
		report.gap_class_ms_per_frame[gap]      = gap_ns[gap] / 1e6 * per_frame;
		report.gap_class_buffers_per_frame[gap] = gap_buffers[gap] * per_frame;
		report.gap_us_per_buffer[gap]           = mean(gap_ns[gap], gap_buffers[gap]) / 1000.0;
		report.gap_us_per_undry_buffer[gap]     = mean(gap_ns[gap], undry) / 1000.0;
	}
	report.calibrated   = after.calibrated;
	report.deviation_us = static_cast<double>(after.calibration_deviation_ns) / 1000.0;
	report.task_binds_per_frame = delta(before.task_binds, after.task_binds) * per_frame;
	report.first_task_s = after.first_task_ns != 0 ? static_cast<double>(after.first_task_ns - 1) / 1e9 : -1.0;
	report.window_end_s =
	    after.thread_start_ns != 0 ? static_cast<double>(after.clock_ns - after.thread_start_ns) / 1e9 : 0.0;
	report.pool_buffers                 = static_cast<double>(after.pool_buffers);
	report.pool_marked                  = static_cast<double>(after.pool_marked);
	report.guest_locks_per_frame        = delta(before.guest_locks, after.guest_locks) * per_frame;
	report.guest_lock_ms_per_frame      = delta(before.guest_lock_ticks, after.guest_lock_ticks) * ms_per_tick * per_frame;
	report.guest_lock_waits_per_frame   = delta(before.guest_lock_waits, after.guest_lock_waits) * per_frame;
	report.guest_lock_wait_ms_per_frame =
	    delta(before.guest_lock_wait_ticks, after.guest_lock_wait_ticks) * ms_per_tick * per_frame;
	report.guest_lock_wait_us = static_cast<double>(GUEST_LOCK_WAIT_TICKS) * us_per_tick;
	return report;
}

[[nodiscard]] constexpr const char* Name(Cause cause) noexcept {
	constexpr const char* NAMES[CAUSE_COUNT] = {
	    "pm4",           "registers",         "draw",          "programs",
	    "resources",     "textures",          "buffers",       "targets",
	    "bindings",      "barriers",          "record",        "compile",
	    "transfers",     "submit begin",      "submit end",    "submit queue", "upkeep", "foreign",
	    "wait GPU read", "wait GPU SRT",      "wait GPU remap", "wait GPU frame",
	    "wait priority", "wait GPU other",    "wait queue lock", "wait label GPU", "wait label CPU",
	    "idle (suspend)", "idle",             "other"};
	return NAMES[static_cast<size_t>(cause)];
}

[[nodiscard]] constexpr const char* Name(Event event) noexcept {
	constexpr const char* NAMES[EVENT_COUNT] = {
	    "packets", "draws",       "dispatches", "barriers",    "command buffers", "submits",
	    "retries", "label polls", "label waits GPU", "label waits CPU", "trusted waits", "GPU waits", "allocations", "releases"};
	return NAMES[static_cast<size_t>(event)];
}

[[nodiscard]] constexpr const char* Name(Close close) noexcept {
	constexpr const char* NAMES[CLOSE_COUNT] = {"GPU idle", "size", "boundary", "before idle", "wait",
	                                            "suspend point", "flip", "CPU flip", "wait flip done", "other"};
	return NAMES[static_cast<size_t>(close)];
}

[[nodiscard]] constexpr const char* Name(TransitCaller caller) noexcept {
	constexpr const char* NAMES[TRANSIT_CALLER_COUNT] = {"targets", "bindings", "heap", "other"};
	return NAMES[static_cast<size_t>(caller)];
}

[[nodiscard]] constexpr const char* Name(Timed step) noexcept {
	constexpr const char* NAMES[TIMED_COUNT] = {"transit, whole image", "transit, per subresource", "draw state"};
	return NAMES[static_cast<size_t>(step)];
}

[[nodiscard]] constexpr const char* Name(Writer writer) noexcept {
	constexpr const char* NAMES[WRITER_COUNT] = {"open", "submitted", "executed"};
	return NAMES[static_cast<size_t>(writer)];
}

[[nodiscard]] constexpr const char* Name(SubmitKind kind) noexcept {
	constexpr const char* NAMES[SUBMIT_KIND_COUNT] = {"task", "marked", "ordinary"};
	return NAMES[static_cast<size_t>(kind)];
}

[[nodiscard]] constexpr const char* Name(SubmitSize size) noexcept {
	constexpr const char* NAMES[SUBMIT_SIZE_COUNT] = {"under 64", "64 to 512", "over 512"};
	return NAMES[static_cast<size_t>(size)];
}

[[nodiscard]] constexpr const char* Name(GpuGap gap) noexcept {
	constexpr const char* NAMES[GPU_GAP_COUNT] = {"dry before the call", "dry by its return", "queued"};
	return NAMES[static_cast<size_t>(gap)];
}

// The cause a group of word waits is booked under (WORD_WAIT_CAUSES).
[[nodiscard]] constexpr const char* WordWaitCauseName(size_t group) noexcept {
	constexpr const char* NAMES[WORD_WAIT_CAUSES] = {"SRT", "read", "another cause"};
	return NAMES[group];
}

} // namespace Common::CommandThreadProfile

#endif /* KYTY_COMMON_COMMAND_THREAD_PROFILE_H_ */
