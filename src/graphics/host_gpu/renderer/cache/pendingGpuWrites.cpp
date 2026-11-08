#include "graphics/host_gpu/renderer/cache/pendingGpuWrites.h"

#include <algorithm>

namespace Libs::Graphics {

std::vector<PendingGpuWrites::Range>::const_iterator
PendingGpuWrites::FirstEndingAfter(uint64_t address) const noexcept {
	// Disjoint and sorted by start, the ranges are sorted by end as well.
	return std::upper_bound(m_ranges.begin(), m_ranges.end(), address,
	                        [](uint64_t value, const Range& range) { return value < range.end; });
}

void PendingGpuWrites::Add(uint64_t address, uint64_t size, uint64_t tick) {
	if (size == 0) {
		return;
	}
	const uint64_t end = address + size;
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
	std::erase_if(m_ranges, [done_tick](const Range& range) { return range.tick <= done_tick; });
}

const PendingGpuWrites::Range* PendingGpuWrites::Find(uint64_t address) const noexcept {
	const auto it = FirstEndingAfter(address);
	return it != m_ranges.end() && it->start <= address ? &*it : nullptr;
}

bool PendingGpuWrites::Intersects(uint64_t address, uint64_t size) const noexcept {
	if (size == 0) {
		return false;
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

} // namespace Libs::Graphics
