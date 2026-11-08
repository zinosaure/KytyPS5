#include "graphics/host_gpu/renderer/occlusionQueries.h"

#include "gpu_tiler_shaders/occlusion_dumps_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <cstring>

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

	m_graphics    = &graphics;
	m_gpu_results = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::DeviceLocal, 0,
	                                         AllFlags, 2ull * QUERY_COUNT * sizeof(uint32_t));
	m_gpu_running = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Download, 0,
	                                         AllFlags, sizeof(uint64_t));
	std::memset(m_gpu_running->Mapped().data(), 0, sizeof(uint64_t));
	m_gpu_running->Flush(0, sizeof(uint64_t));
	SetVulkanObjectNameF(m_device, m_gpu_results->Handle(), "Occlusion Query Results");
	SetVulkanObjectNameF(m_device, m_gpu_running->Handle(), "Occlusion Running Count");
	const vk::DescriptorSetLayoutBinding bindings[] {
	    {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	    {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	    {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	};
	vk::DescriptorSetLayoutCreateInfo layout_info {};
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	layout_info.bindingCount = std::size(bindings);
	layout_info.pBindings    = bindings;
	RequireVulkanSuccess(
	    m_device.createDescriptorSetLayout(&layout_info, nullptr, &m_dump_descriptors),
	    "create occlusion-dump descriptor layout");
	const vk::PushConstantRange push {vk::ShaderStageFlagBits::eCompute, 0, 5 * sizeof(uint32_t)};
	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount         = 1;
	pipeline_layout_info.pSetLayouts            = &m_dump_descriptors;
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push;
	RequireVulkanSuccess(
	    m_device.createPipelineLayout(&pipeline_layout_info, nullptr, &m_dump_layout),
	    "create occlusion-dump pipeline layout");
	const auto module = CompileSPV(OCCLUSION_DUMPS_SPV, m_device);
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
	pipeline_info.stage.module = module;
	pipeline_info.stage.pName  = "main";
	pipeline_info.layout       = m_dump_layout;
	const auto result =
	    m_device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &m_dump_pipeline);
	m_device.destroyShaderModule(module, nullptr);
	RequireVulkanSuccess(result, "create occlusion-dump pipeline");
	SetVulkanObjectNameF(m_device, m_dump_pipeline, "Occlusion Dumps");
}

OcclusionQueries::~OcclusionQueries() {
	if (m_pool) {
		m_device.destroyPipeline(m_dump_pipeline, nullptr);
		m_device.destroyPipelineLayout(m_dump_layout, nullptr);
		m_device.destroyDescriptorSetLayout(m_dump_descriptors, nullptr);
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

// Each dump written by the GPU as Publish writes it, in command order: the GPU keeps the running
// count, adds the samples of the host queries since the last dump (one per draw while a guest query
// is open), and writes the counters. Begin/end pairs, nested queries and lone dumps alike.
bool OcclusionQueries::PublishOnGpu(BufferCache& buffers) {
	if (m_dumps.empty() || !buffers.IsUnified()) {
		return false;
	}
	// A dump's counters span 256 bytes from a begin dump's address.
	constexpr uint64_t span = OCCLUSION_QUERY_DBS * 16u;
	std::vector<std::pair<Buffer*, uint64_t>> targets;
	targets.reserve(m_dumps.size());
	for (const auto& dump: m_dumps) {
		targets.push_back(buffers.UnifiedGpuWriteTarget(dump.address, span));
		if (targets.back().first == nullptr) {
			return false;
		}
	}
	m_gpu_used = true;
	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto native = command.Handle();
	const auto last   = m_dumps.back().queries;
	const auto start  = static_cast<uint32_t>(m_read % QUERY_COUNT);
	const auto total  = static_cast<uint32_t>(last - m_read);
	const auto head   = std::min<uint32_t>(total, QUERY_COUNT - start);
	// The copies wait for the draws' queries, and land after the previous dumps read the results.
	vk::MemoryBarrier before {};
	before.srcAccessMask = vk::AccessFlagBits::eShaderRead;
	before.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 1, &before, 0, nullptr, 0,
	                       nullptr);
	if (head != 0) {
		native.copyQueryPoolResults(m_pool, start, head, m_gpu_results->Handle(),
		                            start * sizeof(uint32_t), sizeof(uint32_t),
		                            vk::QueryResultFlagBits::eWait);
	}
	if (head < total) {
		native.copyQueryPoolResults(m_pool, 0, total - head, m_gpu_results->Handle(),
		                            QUERY_COUNT * sizeof(uint32_t), sizeof(uint32_t),
		                            vk::QueryResultFlagBits::eWait);
	}
	// The counters land after everything recorded before, as the console's end-of-pipe dump does.
	vk::MemoryBarrier copied {};
	copied.srcAccessMask = vk::AccessFlagBits::eTransferWrite | vk::AccessFlagBits::eMemoryWrite;
	copied.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eComputeShader, {}, 1, &copied, 0, nullptr, 0,
	                       nullptr);
	native.bindPipeline(vk::PipelineBindPoint::eCompute, m_dump_pipeline);
	const auto alignment = std::max<uint64_t>(
	    m_graphics->physical_device_properties.limits.minStorageBufferOffsetAlignment, 8);
	uint64_t read = m_read;
	for (size_t i = 0; i < m_dumps.size(); i++) {
		const auto& dump              = m_dumps[i];
		const auto [counters, offset] = targets[i];
		const auto base               = offset / alignment * alignment;
		const vk::DescriptorBufferInfo infos[] {
		    {m_gpu_results->Handle(), 0, m_gpu_results->Size()},
		    {m_gpu_running->Handle(), 0, sizeof(uint64_t)},
		    {counters->Handle(), base, offset - base + span},
		};
		std::array<vk::WriteDescriptorSet, 3> writes {};
		for (uint32_t index = 0; index < writes.size(); ++index) {
			writes[index].dstBinding      = index;
			writes[index].descriptorCount = 1;
			writes[index].descriptorType  = vk::DescriptorType::eStorageBuffer;
			writes[index].pBufferInfo     = &infos[index];
		}
		// The results of [read, dump.queries) are contiguous from read's place in the pool.
		const std::array<uint32_t, 5> params {
		    static_cast<uint32_t>(start + (read - m_read)),
		    static_cast<uint32_t>(dump.queries - read),
		    static_cast<uint32_t>((offset - base) / sizeof(uint64_t)),
		    (dump.address & 8u) == 0 ? 1u : 0u,
		    static_cast<uint32_t>(dump.uncounted),
		};
		native.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_dump_layout, 0, writes);
		native.pushConstants(m_dump_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(params),
		                     params.data());
		native.dispatch(1, 1, 1);
		// The next dump adds to this one's running count.
		vk::MemoryBarrier chained {};
		chained.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
		chained.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
		native.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
		                       vk::PipelineStageFlagBits::eComputeShader, {}, 1, &chained, 0,
		                       nullptr, 0, nullptr);
		read = dump.queries;
	}
	vk::MemoryBarrier written {};
	written.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
	written.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
	                        vk::AccessFlagBits::eHostRead;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
	                       vk::PipelineStageFlagBits::eAllCommands | vk::PipelineStageFlagBits::eHost,
	                       {}, 1, &written, 0, nullptr, 0, nullptr);
	if (head != 0) {
		native.resetQueryPool(m_pool, start, head);
	}
	if (head < total) {
		native.resetQueryPool(m_pool, 0, total - head);
	}
	m_read = last;
	m_dumps.clear();
	return true;
}

uint64_t OcclusionQueries::GpuSamples() const {
	m_gpu_running->Invalidate(0, sizeof(uint64_t));
	uint64_t samples = 0;
	std::memcpy(&samples, m_gpu_running->Mapped().data(), sizeof(samples));
	return samples;
}

void OcclusionQueries::PublishDumps() {
	// The GPU's running count, once PublishOnGpu kept it, is the one to carry on.
	if (m_gpu_used) {
		m_scheduler.Wait(m_dumps.back().tick);
		m_samples = GpuSamples();
	}
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
	if (m_gpu_used) {
		// Later GPU dumps go on from here.
		auto& command = m_scheduler.Current();
		command.EndRendering();
		command.Handle().updateBuffer(m_gpu_running->Handle(), 0, sizeof(m_samples), &m_samples);
		vk::MemoryBarrier updated {};
		updated.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		updated.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
		command.Handle().pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                                 vk::PipelineStageFlagBits::eComputeShader, {}, 1, &updated,
		                                 0, nullptr, 0, nullptr);
	}
}

} // namespace Libs::Graphics
