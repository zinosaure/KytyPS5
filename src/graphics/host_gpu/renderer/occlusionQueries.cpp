#include "graphics/host_gpu/renderer/occlusionQueries.h"

#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"

#include <algorithm>
#include <array>

namespace Libs::Graphics {

// Queries no dump has read yet; past them a draw counts one sample.
constexpr uint32_t QUERY_COUNT = 16384;

OcclusionQueries::OcclusionQueries(GraphicContext& graphics, CommandScheduler& scheduler)
    : m_scheduler(scheduler) {
	if (!graphics.occlusion_queries_enabled) {
		return;
	}
	m_device = graphics.device;
	const vk::QueryPoolCreateInfo info {.queryType  = vk::QueryType::eOcclusion,
	                                    .queryCount = QUERY_COUNT};
	RequireVulkanSuccess(m_device.createQueryPool(&info, nullptr, &m_pool), "vkCreateQueryPool");
	m_device.resetQueryPool(m_pool, 0, QUERY_COUNT);
	m_results.resize(QUERY_COUNT);
	if (graphics.occlusion_query_precise_enabled) {
		m_control = vk::QueryControlFlagBits::ePrecise;
	}
}

OcclusionQueries::~OcclusionQueries() {
	if (m_pool) {
		m_device.destroyQueryPool(m_pool, nullptr);
	}
}

int32_t OcclusionQueries::BeginQuery(vk::CommandBuffer command) {
	if (m_begun - m_read == QUERY_COUNT) {
		m_uncounted++;
		return -1;
	}
	const auto query = static_cast<uint32_t>(m_begun++ % QUERY_COUNT);
	command.beginQuery(m_pool, query, m_control);
	return static_cast<int32_t>(query);
}

void OcclusionQueries::Dump(uint64_t address) {
	// Begins dump to the results, ends 8 bytes into them.
	if ((address & 8u) == 0) {
		m_open++;
	} else if (m_open != 0) {
		m_open--;
	}
	m_dumps.push_back({address, m_scheduler.CurrentTick(), m_begun, m_uncounted});
}

// A guest occlusion query is a begin dump, the draws it counts, an end dump. With one host query
// between them and none older unread, the GPU writes the pair as the console's GPU does: every
// counter ready at zero, then that query's samples in DB 0's end counter. The guest reads
// end - begin, so the pair need not carry the running count.
bool OcclusionQueries::PublishOnGpu(BufferCache& buffers) {
	if (m_dumps.size() != 2u || !buffers.IsUnified()) {
		return false;
	}
	const auto& begin = m_dumps[0];
	const auto& end   = m_dumps[1];
	if ((begin.address & 8u) != 0 || end.address != begin.address + 8u || begin.queries != m_read ||
	    end.queries != begin.queries + 1u || end.uncounted != begin.uncounted) {
		return false;
	}
	std::array<uint64_t, OCCLUSION_QUERY_DBS * 2> counters {};
	counters.fill(OCCLUSION_QUERY_READY);
	if (!buffers.WriteGuestMemory(begin.address, counters.data(), sizeof(counters))) {
		return false;
	}
	const auto query = static_cast<uint32_t>(begin.queries % QUERY_COUNT);
	EXIT_NOT_IMPLEMENTED(!buffers.WriteQueryResult(m_pool, query, end.address));
	m_scheduler.Current().Handle().resetQueryPool(m_pool, query, 1);
	m_read = end.queries;
	m_dumps.clear();
	return true;
}

void OcclusionQueries::PublishDumps() {
	if (const auto last = m_dumps.back().queries; last != m_read) {
		m_scheduler.Wait(m_dumps.back().tick);
		// The unread queries wrap around the pool at most once.
		for (uint64_t read = m_read; read < last;) {
			const auto first = static_cast<uint32_t>(read % QUERY_COUNT);
			const auto count =
			    static_cast<uint32_t>(std::min<uint64_t>(last - read, QUERY_COUNT - first));
			RequireVulkanSuccess(
			    m_device.getQueryPoolResults(m_pool, first, count, count * sizeof(uint64_t),
				                             m_results.data() + first, sizeof(uint64_t),
				                             vk::QueryResultFlagBits::e64),
			    "vkGetQueryPoolResults");
			m_device.resetQueryPool(m_pool, first, count);
			read += count;
		}
	}
	for (const auto& dump: m_dumps) {
		for (; m_read < dump.queries; m_read++) {
			m_samples += m_results[m_read % QUERY_COUNT];
		}
		WriteZPassCounters(reinterpret_cast<volatile uint64_t*>(dump.address),
		                   m_samples + dump.uncounted, (dump.address & 8u) == 0);
	}
	m_dumps.clear();
}

} // namespace Libs::Graphics
