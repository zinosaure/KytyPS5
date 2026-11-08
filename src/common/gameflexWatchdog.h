#ifndef EMULATOR_INCLUDE_COMMON_GAMEFLEX_WATCHDOG_H_
#define EMULATOR_INCLUDE_COMMON_GAMEFLEX_WATCHDOG_H_

// GameFlex: who waits when a game stops showing images. A game that hangs on a black screen writes
// nothing (Kingdom Hearts III on the phone, after its logo video, #632). Every blocking point of the
// game's threads, and of Kyty's presentation and GPU threads, notes what it waits for (WaitNote: three
// relaxed stores, no clock), with the game's code it waits from, and a watchdog thread prints them once the game
// has shown no new image for 10 s, then once more 30 s later. KYTY_WATCHDOG=0 turns the watchdog off.
// Every call of the console's libraries (Kyty's functions behind the game's imports, through PRINT_NAME) is noted
// too (CallNote): with each thread's processor time, the dump says which thread runs, what it calls and from where in
// the game, and which one is stuck inside a call; a thread that runs without calling anything is sampled by a signal,
// which gives the address of the game's code it turns in.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <csignal>
#include <dirent.h>
#include <pthread.h>
#include <sys/syscall.h>
#include <ucontext.h>
#include <unistd.h>

namespace Common::GameFlex {

// One call of the console's libraries: the function, and the game's code it was called from.
struct CallRecord {
	std::atomic<const char*> function {nullptr};
	std::atomic<uint64_t>    caller {0};
};
inline constexpr uint32_t CALL_HISTORY = 16;

struct WaitSlot {
	std::atomic<const char*> what {nullptr};
	std::atomic<uint64_t>    object {0};
	// Where the guest called from (its return address): the game's code that waits, for an attente du jeu.
	std::atomic<uint64_t>    caller {0};
	std::atomic<uint32_t>    sequence {0};
	long                     tid = 0;
	// The watchdog's own view: the sequence it saw last, and for how many seconds it has not moved.
	uint32_t seen_sequence = 0;
	uint32_t seconds       = 0;
	// The calls of the console's libraries: how many, the last ones, how deep the thread is inside them now, and the
	// innermost one it is in.
	std::atomic<uint32_t>    calls {0};
	std::atomic<uint32_t>    inside {0};
	std::atomic<const char*> call {nullptr};
	std::atomic<uint64_t>    call_caller {0};
	CallRecord               history[CALL_HISTORY];
	// The watchdog's view of them: the count it saw last, for how many seconds the thread has stayed inside a call
	// without making another, and the count when the activity window began.
	uint32_t seen_calls   = 0;
	uint32_t call_seconds = 0;
	uint32_t window_calls = 0;
};

inline std::mutex             g_wait_slots_mutex;
inline std::vector<WaitSlot*> g_wait_slots; // One per thread that ever waited; never freed.

inline WaitSlot& ThisThreadWaitSlot() {
	thread_local WaitSlot* slot = [] {
		auto* created = new WaitSlot;
		created->tid  = syscall(SYS_gettid);
		std::lock_guard lock(g_wait_slots_mutex);
		g_wait_slots.push_back(created);
		return created;
	}();
	return *slot;
}

// What this thread waits for, while it waits. A nested note restores the outer one.
class WaitNote {
public:
	explicit WaitNote(const char* what, uint64_t object = 0, uint64_t caller = 0): m_slot(ThisThreadWaitSlot()) {
		m_previous        = m_slot.what.load(std::memory_order_relaxed);
		m_previous_object = m_slot.object.load(std::memory_order_relaxed);
		m_previous_caller = m_slot.caller.load(std::memory_order_relaxed);
		m_slot.object.store(object, std::memory_order_relaxed);
		m_slot.caller.store(caller, std::memory_order_relaxed);
		m_slot.sequence.store(m_slot.sequence.load(std::memory_order_relaxed) + 1,
		                      std::memory_order_relaxed);
		m_slot.what.store(what, std::memory_order_relaxed);
	}
	template <typename T>
	WaitNote(const char* what, T* object, const void* caller = nullptr)
	    : WaitNote(what, reinterpret_cast<uint64_t>(object), reinterpret_cast<uint64_t>(caller)) {}
	~WaitNote() {
		m_slot.caller.store(m_previous_caller, std::memory_order_relaxed);
		m_slot.object.store(m_previous_object, std::memory_order_relaxed);
		m_slot.what.store(m_previous, std::memory_order_relaxed);
	}
	WaitNote(const WaitNote&)            = delete;
	WaitNote& operator=(const WaitNote&) = delete;

private:
	WaitSlot&   m_slot;
	const char* m_previous        = nullptr;
	uint64_t    m_previous_object = 0;
	uint64_t    m_previous_caller = 0;
};

// A call of the console's libraries, while it runs: noted by PRINT_NAME at the top of each of Kyty's functions.
class CallNote {
public:
	CallNote(const char* function, const void* caller): m_slot(ThisThreadWaitSlot()) {
		const auto count  = m_slot.calls.load(std::memory_order_relaxed);
		auto&      record = m_slot.history[count % CALL_HISTORY];
		record.function.store(function, std::memory_order_relaxed);
		record.caller.store(reinterpret_cast<uint64_t>(caller), std::memory_order_relaxed);
		m_previous        = m_slot.call.load(std::memory_order_relaxed);
		m_previous_caller = m_slot.call_caller.load(std::memory_order_relaxed);
		m_slot.call_caller.store(reinterpret_cast<uint64_t>(caller), std::memory_order_relaxed);
		m_slot.call.store(function, std::memory_order_relaxed);
		m_slot.inside.store(m_slot.inside.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
		m_slot.calls.store(count + 1, std::memory_order_relaxed);
	}
	~CallNote() {
		// A fiber may come back on another thread: that thread's slot is not ours to change.
		if (&ThisThreadWaitSlot() != &m_slot) {
			return;
		}
		m_slot.inside.store(m_slot.inside.load(std::memory_order_relaxed) - 1, std::memory_order_relaxed);
		m_slot.call_caller.store(m_previous_caller, std::memory_order_relaxed);
		m_slot.call.store(m_previous, std::memory_order_relaxed);
	}
	CallNote(const CallNote&)            = delete;
	CallNote& operator=(const CallNote&) = delete;

private:
	WaitSlot&   m_slot;
	const char* m_previous        = nullptr;
	uint64_t    m_previous_caller = 0;
};

// Names an address of the game's code (« eboot.bin+0x1234 ») : set by the video out, which can ask the loader.
inline std::string (*g_describe_address)(uint64_t address) = nullptr;

inline std::string DescribeAddress(uint64_t address) {
	if (address == 0) {
		return {};
	}
	if (g_describe_address != nullptr) {
		return g_describe_address(address);
	}
	char text[32];
	std::snprintf(text, sizeof(text), "0x%" PRIx64, address);
	return text;
}

inline uint64_t SteadyMs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

// The flips: reserved by the game, presented by the present thread, and what the present thread's
// last look at the flip queue found.
enum FlipQueueState : uint32_t {
	FlipQueueUnseen,
	FlipQueueEmpty,
	FlipQueueNotReady, // The oldest flip waits for its GPU work or its preparation.
	FlipQueueNotDue,   // Its vblank has not come (flip rate).
	FlipQueuePresented,
};
inline std::atomic<uint64_t> g_flips_reserved {0};
inline std::atomic<uint64_t> g_flips_presented {0};
inline std::atomic<uint32_t> g_flip_queue_state {FlipQueueUnseen};
inline std::atomic<uint32_t> g_flip_queue_requests {0};
inline std::atomic<uint32_t> g_flip_queue_cpu_requests {0};

inline void NoteFlipQueue(FlipQueueState state, size_t requests, size_t cpu_requests) {
	g_flip_queue_requests.store(static_cast<uint32_t>(requests), std::memory_order_relaxed);
	g_flip_queue_cpu_requests.store(static_cast<uint32_t>(cpu_requests), std::memory_order_relaxed);
	g_flip_queue_state.store(state, std::memory_order_relaxed);
}

// The guest GPU: since when it has completed no submission while one waits, and what the last
// suspended command waited for.
inline std::atomic<uint64_t>    g_gpu_blocked_since_ms {0};
inline std::atomic<const char*> g_gpu_wait_kind {nullptr};
inline std::atomic<uint64_t>    g_gpu_wait_address {0};
inline std::atomic<uint64_t>    g_gpu_wait_value {0};
inline std::atomic<uint64_t>    g_gpu_wait_reference {0};
inline std::atomic<uint64_t>    g_gpu_wait_mask {0};
inline std::atomic<uint32_t>    g_gpu_wait_function {0};

inline void NoteGpuWait(const char* kind, uint64_t address, uint64_t value, uint64_t reference,
                        uint64_t mask, uint32_t function) {
	g_gpu_wait_address.store(address, std::memory_order_relaxed);
	g_gpu_wait_value.store(value, std::memory_order_relaxed);
	g_gpu_wait_reference.store(reference, std::memory_order_relaxed);
	g_gpu_wait_mask.store(mask, std::memory_order_relaxed);
	g_gpu_wait_function.store(function, std::memory_order_relaxed);
	g_gpu_wait_kind.store(kind, std::memory_order_relaxed);
	if (g_gpu_blocked_since_ms.load(std::memory_order_relaxed) == 0) {
		g_gpu_blocked_since_ms.store(SteadyMs(), std::memory_order_relaxed);
	}
}

inline void NoteGpuProgress() {
	if (g_gpu_blocked_since_ms.load(std::memory_order_relaxed) != 0) {
		g_gpu_blocked_since_ms.store(0, std::memory_order_relaxed);
	}
}

inline std::string ThreadName(long tid) {
	char path[64];
	std::snprintf(path, sizeof(path), "/proc/self/task/%ld/comm", tid);
	std::FILE* file = std::fopen(path, "r");
	if (file == nullptr) {
		return {};
	}
	char name[64] = {};
	if (std::fgets(name, sizeof(name), file) == nullptr) {
		name[0] = '\0';
	}
	std::fclose(file);
	name[std::strcspn(name, "\n")] = '\0';
	return name;
}

// A thread's state and processor time, from /proc (ticks of user and system time).
inline bool ReadThreadTicks(long tid, char* state, uint64_t* ticks) {
	char path[64];
	std::snprintf(path, sizeof(path), "/proc/self/task/%ld/stat", tid);
	std::FILE* file = std::fopen(path, "r");
	if (file == nullptr) {
		return false;
	}
	char         text[1024] = {};
	const size_t size       = std::fread(text, 1, sizeof(text) - 1, file);
	std::fclose(file);
	text[size]        = '\0';
	const char* close = std::strrchr(text, ')');
	if (close == nullptr || close[1] == '\0') {
		return false;
	}
	unsigned long long user   = 0;
	unsigned long long system = 0;
	// After the name: the state, ten fields, then the user and system times.
	if (std::sscanf(close + 2, "%c %*d %*d %*d %*d %*d %*u %*u %*u %*u %*u %llu %llu", state, &user, &system) != 3) {
		return false;
	}
	*ticks = user + system;
	return true;
}

inline std::vector<long> ProcessThreads() {
	std::vector<long> tids;
	DIR*              dir = opendir("/proc/self/task");
	if (dir == nullptr) {
		return tids;
	}
	while (const auto* entry = readdir(dir)) {
		const long tid = std::strtol(entry->d_name, nullptr, 10);
		if (tid > 0) {
			tids.push_back(tid);
		}
	}
	closedir(dir);
	return tids;
}

// The start of the activity window: each thread's processor time, and each slot's calls. The watchdog thread only.
inline std::map<long, uint64_t> g_activity_ticks;
inline uint64_t                 g_activity_since_ms = 0;

inline void SnapshotActivity() {
	g_activity_ticks.clear();
	for (const long tid: ProcessThreads()) {
		char     state = '?';
		uint64_t ticks = 0;
		if (ReadThreadTicks(tid, &state, &ticks)) {
			g_activity_ticks[tid] = ticks;
		}
	}
	{
		std::lock_guard lock(g_wait_slots_mutex);
		for (auto* slot: g_wait_slots) {
			slot->window_calls = slot->calls.load(std::memory_order_relaxed);
		}
	}
	g_activity_since_ms = SteadyMs();
}

// Where a running thread is: a signal stops it for an instant, its handler reads the address of the code it was
// running (under Box64, the x86 address of the guest).
inline std::atomic<long>     g_sample_tid {0};
inline std::atomic<uint64_t> g_sample_address {0};

inline int SampleSignal() {
	return SIGRTMIN + 6; // Kyty's guest signals use SIGRTMIN + 3.
}

inline void OnSampleSignal(int /*signal*/, siginfo_t* /*info*/, void* context) {
	if (syscall(SYS_gettid) != g_sample_tid.load(std::memory_order_relaxed)) {
		return;
	}
#if defined(__x86_64__)
	const auto* registers = static_cast<const ucontext_t*>(context);
	g_sample_address.store(static_cast<uint64_t>(registers->uc_mcontext.gregs[REG_RIP]), std::memory_order_relaxed);
#else
	(void)context;
#endif
}

inline void InstallSampler() {
	struct sigaction action {};
	action.sa_sigaction = OnSampleSignal;
	action.sa_flags     = SA_SIGINFO | SA_RESTART | SA_ONSTACK;
	sigemptyset(&action.sa_mask);
	sigaction(SampleSignal(), &action, nullptr);
}

inline uint64_t SampleAddress(long tid) {
	g_sample_address.store(0, std::memory_order_relaxed);
	g_sample_tid.store(tid, std::memory_order_relaxed);
	if (syscall(SYS_tgkill, getpid(), tid, SampleSignal()) != 0) {
		g_sample_tid.store(0, std::memory_order_relaxed);
		return 0;
	}
	for (int wait = 0; wait < 40 && g_sample_address.load(std::memory_order_relaxed) == 0; wait++) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	g_sample_tid.store(0, std::memory_order_relaxed);
	return g_sample_address.load(std::memory_order_relaxed);
}

// What each thread did since the window began: its processor time, its calls of the console's libraries (the most
// frequent of its last ones, with the game's code they come from), the call it is stuck in; and where the busiest
// threads that call nothing are running.
inline void DumpActivity() {
	const double seconds = std::max(1.0, static_cast<double>(SteadyMs() - g_activity_since_ms) / 1000.0);
	const double hz      = static_cast<double>(sysconf(_SC_CLK_TCK));
	struct Activity {
		long     tid;
		char     state;
		double   cpu;
		uint32_t calls;
		WaitSlot* slot;
	};
	std::map<long, WaitSlot*> slots;
	{
		std::lock_guard lock(g_wait_slots_mutex);
		for (auto* slot: g_wait_slots) {
			slots[slot->tid] = slot;
		}
	}
	std::vector<Activity> threads;
	for (const long tid: ProcessThreads()) {
		char     state = '?';
		uint64_t ticks = 0;
		if (!ReadThreadTicks(tid, &state, &ticks)) {
			continue;
		}
		const auto before = g_activity_ticks.find(tid);
		const double cpu  = before == g_activity_ticks.end() ? 0.0
		                                                     : static_cast<double>(ticks - before->second) / hz / seconds;
		auto*          slot  = slots.count(tid) != 0 ? slots[tid] : nullptr;
		const uint32_t calls = slot != nullptr ? slot->calls.load(std::memory_order_relaxed) - slot->window_calls : 0;
		threads.push_back({tid, state, cpu, calls, slot});
	}
	std::sort(threads.begin(), threads.end(), [](const Activity& a, const Activity& b) {
		return a.cpu != b.cpu ? a.cpu > b.cpu : a.calls > b.calls;
	});
	std::printf("[watchdog] the last %.0f s, thread by thread: processor, calls of the console's libraries\n", seconds);
	int shown   = 0;
	int sampled = 0;
	for (const auto& thread: threads) {
		const bool stuck_inside = thread.slot != nullptr && thread.slot->call_seconds >= 3 &&
		                          thread.slot->inside.load(std::memory_order_relaxed) > 0;
		if (shown >= 12 || (thread.cpu < 0.02 && thread.calls == 0 && !stuck_inside)) {
			continue;
		}
		const auto name = ThreadName(thread.tid);
		if (name.empty()) {
			continue;
		}
		shown++;
		std::string detail;
		if (thread.slot != nullptr && thread.calls > 0) {
			// The most frequent of its last calls, each with the game's code it comes from.
			std::map<std::pair<const char*, uint64_t>, uint32_t> recent;
			const uint32_t total = thread.slot->calls.load(std::memory_order_relaxed);
			for (uint32_t i = 0; i < std::min(total, CALL_HISTORY); i++) {
				const auto& record = thread.slot->history[i];
				const auto* function = record.function.load(std::memory_order_relaxed);
				if (function != nullptr) {
					recent[{function, record.caller.load(std::memory_order_relaxed)}]++;
				}
			}
			std::vector<std::pair<uint32_t, std::pair<const char*, uint64_t>>> ranked;
			for (const auto& [call, count]: recent) {
				ranked.push_back({count, call});
			}
			std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
			for (size_t i = 0; i < std::min<size_t>(ranked.size(), 3); i++) {
				const auto from = DescribeAddress(ranked[i].second.second);
				detail += (i == 0 ? ": " : ", ") + std::string(ranked[i].second.first) + " x" +
				          std::to_string(ranked[i].first) + (from.empty() ? "" : " from " + from);
			}
		}
		if (stuck_inside) {
			const auto* call = thread.slot->call.load(std::memory_order_relaxed);
			const auto  from = DescribeAddress(thread.slot->call_caller.load(std::memory_order_relaxed));
			detail += "; inside " + std::string(call != nullptr ? call : "?") + (from.empty() ? "" : " from " + from) +
			          " for " + std::to_string(thread.slot->call_seconds) + " s";
		}
		std::printf("[watchdog]   %s (%ld) %c %.0f%%, %u calls%s\n", name.c_str(), thread.tid, thread.state,
		            thread.cpu * 100.0, thread.calls, detail.c_str());
		// A busy thread: where it runs, eight looks at it.
		if (thread.cpu >= 0.3 && sampled < 3) {
			sampled++;
			std::map<uint64_t, uint32_t> where;
			for (int look = 0; look < 8; look++) {
				const auto address = SampleAddress(thread.tid);
				if (address != 0) {
					where[address]++;
				}
				std::this_thread::sleep_for(std::chrono::milliseconds(15));
			}
			std::string places;
			for (const auto& [address, count]: where) {
				places += (places.empty() ? "" : ", ") + DescribeAddress(address) + (count > 1 ? " x" + std::to_string(count) : "");
			}
			std::printf("[watchdog]     running at %s\n", places.empty() ? "(no answer to the signal)" : places.c_str());
		}
	}
}

inline void WatchdogDump(uint32_t still) {
	static constexpr const char* STATES[] = {"nothing yet", "an empty queue", "a flip not ready",
	                                         "a flip not due", "a flip presented"};
	const auto state = std::min<uint32_t>(g_flip_queue_state.load(std::memory_order_relaxed), 4);
	std::printf("[watchdog] no new image for %u s: flips reserved %" PRIu64 ", presented %" PRIu64
	            "; the present thread last found %s (requests %u, cpu %u)\n",
	            still, g_flips_reserved.load(std::memory_order_relaxed),
	            g_flips_presented.load(std::memory_order_relaxed), STATES[state],
	            g_flip_queue_requests.load(std::memory_order_relaxed),
	            g_flip_queue_cpu_requests.load(std::memory_order_relaxed));
	const auto since = g_gpu_blocked_since_ms.load(std::memory_order_relaxed);
	const auto* kind = g_gpu_wait_kind.load(std::memory_order_relaxed);
	if (since != 0) {
		std::printf("[watchdog] gpu: no submission done for %" PRIu64
		            " s, the last one waits on %s: *0x%" PRIx64 " = 0x%" PRIx64
		            ", function %u, reference 0x%" PRIx64 ", mask 0x%" PRIx64 "\n",
		            (SteadyMs() - since) / 1000, kind != nullptr ? kind : "?",
		            g_gpu_wait_address.load(std::memory_order_relaxed),
		            g_gpu_wait_value.load(std::memory_order_relaxed),
		            g_gpu_wait_function.load(std::memory_order_relaxed),
		            g_gpu_wait_reference.load(std::memory_order_relaxed),
		            g_gpu_wait_mask.load(std::memory_order_relaxed));
	} else {
		std::printf("[watchdog] gpu: no submission blocked (last wait: %s)\n",
		            kind != nullptr ? kind : "none");
	}
	struct Line {
		uint32_t    seconds;
		long        tid;
		const char* what;
		uint64_t    object;
		uint64_t    caller;
	};
	std::vector<Line> stuck;
	std::vector<Line> brief; // Threads in short, repeated waits (a loop that sleeps and checks again).
	{
		std::lock_guard lock(g_wait_slots_mutex);
		for (auto* slot: g_wait_slots) {
			const auto* what = slot->what.load(std::memory_order_relaxed);
			if (what == nullptr) {
				continue;
			}
			const Line line {slot->seconds, slot->tid, what, slot->object.load(std::memory_order_relaxed),
			                 slot->caller.load(std::memory_order_relaxed)};
			(slot->seconds >= 3 ? stuck : brief).push_back(line);
		}
	}
	std::sort(stuck.begin(), stuck.end(),
	          [](const Line& a, const Line& b) { return a.seconds > b.seconds; });
	// The phone relays 200 lines of a game's output: the longest waits only.
	if (stuck.size() > 12) {
		stuck.resize(12);
	}
	for (const auto& line: stuck) {
		const auto name = ThreadName(line.tid);
		if (name.empty()) {
			continue; // The thread is gone.
		}
		const auto from = DescribeAddress(line.caller);
		std::printf("[watchdog] %s (%ld): %s 0x%" PRIx64 " for %u s%s%s\n", name.c_str(), line.tid, line.what,
		            line.object, line.seconds, from.empty() ? "" : ", from ", from.c_str());
	}
	// The threads that sleep and check again, each by name: what it waits on, and from where in the game.
	std::map<std::string, uint32_t> loops;
	for (const auto& line: brief) {
		const auto name = ThreadName(line.tid);
		if (name.empty()) {
			continue;
		}
		const auto from = DescribeAddress(line.caller);
		loops[name + ": " + line.what + (from.empty() ? "" : " from " + from)]++;
	}
	for (const auto& [loop, count]: loops) {
		std::printf("[watchdog] short wait, %s%s\n", loop.c_str(),
		            count > 1 ? (" (x" + std::to_string(count) + ")").c_str() : "");
	}
	DumpActivity();
	std::fflush(stdout);
}

inline void WatchdogRun() {
	pthread_setname_np(pthread_self(), "GF watchdog");
	uint64_t last_flips = 0;
	uint32_t still      = 0;
	uint32_t dumps      = 0;
	for (;;) {
		std::this_thread::sleep_for(std::chrono::seconds(1));
		{
			std::lock_guard lock(g_wait_slots_mutex);
			for (auto* slot: g_wait_slots) {
				const auto sequence = slot->sequence.load(std::memory_order_relaxed);
				if (slot->what.load(std::memory_order_relaxed) != nullptr &&
				    sequence == slot->seen_sequence) {
					slot->seconds++;
				} else {
					slot->seconds       = 0;
					slot->seen_sequence = sequence;
				}
				const auto calls = slot->calls.load(std::memory_order_relaxed);
				if (slot->inside.load(std::memory_order_relaxed) > 0 && calls == slot->seen_calls) {
					slot->call_seconds++;
				} else {
					slot->call_seconds = 0;
					slot->seen_calls   = calls;
				}
			}
		}
		const auto flips = g_flips_presented.load(std::memory_order_relaxed);
		if (flips != last_flips) {
			last_flips = flips;
			still      = 0;
			dumps      = 0;
			continue;
		}
		if (flips == 0) {
			continue; // Not started yet.
		}
		still++;
		if (still == 1) {
			SnapshotActivity();
		}
		if ((still == 10 || still % 30 == 0) && dumps < 2) {
			dumps++;
			WatchdogDump(still);
			SnapshotActivity();
		}
	}
}

inline void StartWatchdog() {
	const char* setting = std::getenv("KYTY_WATCHDOG");
	if (setting != nullptr && std::strcmp(setting, "0") == 0) {
		return;
	}
	static std::once_flag once;
	std::call_once(once, [] {
		InstallSampler();
		std::thread(WatchdogRun).detach();
	});
}

} // namespace Common::GameFlex

#endif /* EMULATOR_INCLUDE_COMMON_GAMEFLEX_WATCHDOG_H_ */
