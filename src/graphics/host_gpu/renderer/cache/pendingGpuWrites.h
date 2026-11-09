#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_PENDINGGPUWRITES_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_PENDINGGPUWRITES_H_

#include <cstdint>
#include <vector>

namespace Libs::Graphics {

// The guest ranges the GPU may still be writing: bound for writing in a command buffer the GPU has not finished.
// Sorted by address, disjoint (overlapping and touching ranges merge), each with the tick of the last buffer
// that writes it. Every read of a shader's table of resources asks whether its word lies in one: a binary search,
// where a std::map scanned from the first range in reach cost 7 % of the command thread in Astro Bot.
class PendingGpuWrites {
public:
	struct Range {
		uint64_t start;
		uint64_t end;  // one past the last byte
		uint64_t tick; // the command buffer that writes it last
	};

	void Add(uint64_t address, uint64_t size, uint64_t tick);
	// Forgets the ranges whose last writer the GPU has finished.
	void Expire(uint64_t done_tick);
	void Clear() noexcept { m_ranges.clear(); }

	[[nodiscard]] bool         Empty() const noexcept { return m_ranges.empty(); }
	[[nodiscard]] const Range* Find(uint64_t address) const noexcept;
	[[nodiscard]] bool         Intersects(uint64_t address, uint64_t size) const noexcept;
	// How many bytes of [address, address + size) the ranges cover.
	[[nodiscard]] uint64_t     Bytes(uint64_t address, uint64_t size) const noexcept;

private:
	// The first range that ends after `address`.
	[[nodiscard]] std::vector<Range>::const_iterator FirstEndingAfter(uint64_t address) const noexcept;

	std::vector<Range> m_ranges;
	// The GPU tick of the last expiry: ranges added since carry later ticks, so the same tick finds nothing.
	uint64_t           m_expired_tick = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_PENDINGGPUWRITES_H_
