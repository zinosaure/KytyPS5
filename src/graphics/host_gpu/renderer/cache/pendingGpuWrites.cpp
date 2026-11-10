#include "graphics/host_gpu/renderer/cache/pendingGpuWrites.h"

#include <algorithm>

namespace Libs::Graphics {

std::vector<PendingGpuWrites::Range>::const_iterator
PendingGpuWrites::FirstEndingAfter(uint64_t address) const noexcept {
	// Disjoint and sorted by start, the ranges are sorted by end as well.
	return std::upper_bound(m_ranges.begin(), m_ranges.end(), address,
	                        [](uint64_t value, const Range& range) { return value < range.end; });
}

void PendingGpuWrites::MarkBlocks(uint64_t start, uint64_t end) noexcept {
	const auto first = start >> BlockBits;
	const auto last  = (end - 1) >> BlockBits;
	if (last - first >= BlockCount - 1) {
		m_blocks.fill(~uint64_t {0});
		return;
	}
	for (auto block = first; block <= last; block++) {
		const auto bit = block % BlockCount;
		m_blocks[bit / 64] |= uint64_t {1} << (bit % 64);
	}
}

void PendingGpuWrites::Clear() noexcept {
	m_ranges.clear();
	m_blocks.fill(0);
}

void PendingGpuWrites::Add(uint64_t address, uint64_t size, uint64_t tick) {
	if (size == 0) {
		return;
	}
	const uint64_t end = address + size;
	MarkBlocks(address, end);
	// The first range that ends at or after the new one starts: it touches or overlaps it, or comes after.
	auto first = std::lower_bound(m_ranges.begin(), m_ranges.end(), address,
	                              [](const Range& range, uint64_t value) { return range.end < value; });
	if (first == m_ranges.end() || first->start > end) {
		m_ranges.insert(first, {address, end, tick});
		return;
	}
	Range merged {std::min(first->start, address), std::max(first->end, end), std::max(first->tick, tick)};
	auto  last = first + 1;
	while (last != m_ranges.end() && last->start <= merged.end) {
		merged.end  = std::max(merged.end, last->end);
		merged.tick = std::max(merged.tick, last->tick);
		++last;
	}
	*first = merged;
	m_ranges.erase(first + 1, last);
}

void PendingGpuWrites::Expire(uint64_t done_tick) {
	// Asked at every resource-table read and guest wait, mostly with the same tick: a scan only when it moved.
	if (done_tick == m_expired_tick) {
		return;
	}
	m_expired_tick = done_tick;
	const auto before = m_ranges.size();
	std::erase_if(m_ranges, [done_tick](const Range& range) { return range.tick <= done_tick; });
	if (m_ranges.size() == before) {
		return;
	}
	m_blocks.fill(0);
	for (const auto& range: m_ranges) {
		MarkBlocks(range.start, range.end);
	}
}

const PendingGpuWrites::Range* PendingGpuWrites::Find(uint64_t address) const noexcept {
	if (!MaybeTouched(address)) {
		return nullptr;
	}
	const auto it = FirstEndingAfter(address);
	return it != m_ranges.end() && it->start <= address ? &*it : nullptr;
}

bool PendingGpuWrites::Intersects(uint64_t address, uint64_t size) const noexcept {
	if (size == 0) {
		return false;
	}
	// A small read asks the blocks first; a large one goes straight to the search.
	if (const auto first = address >> BlockBits, last = (address + size - 1) >> BlockBits; last - first < 4) {
		bool touched = false;
		for (auto block = first; block <= last && !touched; block++) {
			touched = MaybeTouched(block << BlockBits);
		}
		if (!touched) {
			return false;
		}
	}
	const auto it = FirstEndingAfter(address);
	return it != m_ranges.end() && it->start < address + size;
}

uint64_t PendingGpuWrites::Bytes(uint64_t address, uint64_t size) const noexcept {
	uint64_t bytes = 0;
	for (auto it = FirstEndingAfter(address); it != m_ranges.end() && it->start < address + size; ++it) {
		bytes += std::min(it->end, address + size) - std::max(it->start, address);
	}
	return bytes;
}

uint64_t PendingGpuWrites::LatestTick(uint64_t address, uint64_t size) const noexcept {
	uint64_t tick = 0;
	for (auto it = FirstEndingAfter(address); it != m_ranges.end() && it->start < address + size; ++it) {
		tick = std::max(tick, it->tick);
	}
	return tick;
}

} // namespace Libs::Graphics
