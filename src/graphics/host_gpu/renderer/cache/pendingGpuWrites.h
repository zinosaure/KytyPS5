#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_PENDINGGPUWRITES_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_PENDINGGPUWRITES_H_

#include <array>
#include <cstdint>
#include <vector>

namespace Libs::Graphics {

// The guest ranges the GPU may still be writing: bound for writing in a command buffer the GPU has not finished.
// Sorted by address, disjoint (overlapping and touching ranges merge), each with the tick of the last buffer
// that writes it. Every read of a shader's table of resources asks whether its word lies in one: a binary search,
// where a std::map scanned from the first range in reach cost 7 % of the command thread in Astro Bot. Once the GPU
// ran alongside the thread (0077), about a thousand labels stayed in flight and the search over them, word by word,
// took 10 % of the thread: a bit per 64 KiB block that a range touches answers most words without the search.
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
	void Clear() noexcept;

	[[nodiscard]] bool         Empty() const noexcept { return m_ranges.empty(); }
	[[nodiscard]] const Range* Find(uint64_t address) const noexcept;
	[[nodiscard]] bool         Intersects(uint64_t address, uint64_t size) const noexcept;
	// How many bytes of [address, address + size) the ranges cover.
	[[nodiscard]] uint64_t     Bytes(uint64_t address, uint64_t size) const noexcept;

private:
	// The first range that ends after `address`.
	[[nodiscard]] std::vector<Range>::const_iterator FirstEndingAfter(uint64_t address) const noexcept;

	static constexpr uint32_t BlockBits  = 16;    // 64 KiB blocks
	static constexpr uint32_t BlockCount = 65536; // a bit each, the block number taken modulo the count

	[[nodiscard]] bool MaybeTouched(uint64_t address) const noexcept {
		const auto block = (address >> BlockBits) % BlockCount;
		return ((m_blocks[block / 64] >> (block % 64)) & 1u) != 0;
	}
	void MarkBlocks(uint64_t start, uint64_t end) noexcept;

	std::vector<Range> m_ranges;
	// A bit set for every block a range touches; others share a bit by the modulo, which only costs the search.
	std::array<uint64_t, BlockCount / 64> m_blocks {};
	// The GPU tick of the last expiry: ranges added since carry later ticks, so the same tick finds nothing.
	uint64_t           m_expired_tick = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_PENDINGGPUWRITES_H_
