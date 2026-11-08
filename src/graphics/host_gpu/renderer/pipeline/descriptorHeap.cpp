#include "graphics/host_gpu/renderer/pipeline/descriptorHeap.h"

#include "common/assert.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include <algorithm>
#include <atomic>
#include <cstdio>

namespace Libs::Graphics {
namespace {

constexpr uint32_t   DescriptorHeapCount = 1024;
constexpr std::array DescriptorPoolSizes = {
    vk::DescriptorPoolSize {vk::DescriptorType::eStorageBuffer, 8192},
    vk::DescriptorPoolSize {vk::DescriptorType::eSampledImage, 8192},
    vk::DescriptorPoolSize {vk::DescriptorType::eStorageImage, 1024},
    vk::DescriptorPoolSize {vk::DescriptorType::eSampler, 1024},
};
constexpr std::array<const char*, DescriptorPoolSizes.size()> DescriptorPoolNames = {
    "storage buffer", "sampled image", "storage image", "sampler"};

// Sets of the largest layout a pool must hold, beyond the default quotas.
constexpr uint32_t LargeSetsPerPool = 4;

std::array<std::atomic<uint32_t>, DescriptorPoolSizes.size()> g_required_pool_sizes {};

} // namespace

void RequireDescriptorPoolSize(vk::DescriptorType type, uint32_t count) {
	for (size_t index = 0; index < DescriptorPoolSizes.size(); index++) {
		if (DescriptorPoolSizes[index].type != type) {
			continue;
		}
		const auto needed  = count * LargeSetsPerPool;
		auto       current = g_required_pool_sizes[index].load(std::memory_order_relaxed);
		while (needed > current) {
			if (g_required_pool_sizes[index].compare_exchange_weak(current, needed,
			                                                       std::memory_order_relaxed)) {
				if (count > DescriptorPoolSizes[index].descriptorCount) {
					std::fprintf(stderr,
					             "Descriptor pools grow: a set layout needs %u %s descriptors, "
					             "more than the default %u\n",
					             count, DescriptorPoolNames[index], DescriptorPoolSizes[index].descriptorCount);
				}
				break;
			}
		}
		return;
	}
}

DescriptorHeap::DescriptorHeap(GraphicContext& graphics, MasterSemaphore& master_semaphore)
    : m_graphics(graphics), m_master_semaphore(master_semaphore) {
	CreateDescriptorPool();
}

DescriptorHeap::~DescriptorHeap() {
	m_graphics.device.destroyDescriptorPool(m_current_pool, nullptr);
	for (const auto& [pool, tick]: m_pending_pools) {
		m_master_semaphore.Wait(tick);
		m_graphics.device.destroyDescriptorPool(pool, nullptr);
	}
}

vk::DescriptorSet DescriptorHeap::Commit(vk::DescriptorSetLayout layout) {
	KYTY_PROFILER_FUNCTION();
	auto& batch = m_sets[layout];
	if (batch.size != 0) {
		return batch.sets[--batch.size];
	}
	if (Allocate(layout, batch)) {
		return batch.sets[--batch.size];
	}

	m_pending_pools.emplace_back(m_current_pool, m_master_semaphore.CurrentTick());
	if (const auto& [pool, tick] = m_pending_pools.front(); m_master_semaphore.IsFree(tick)) {
		m_current_pool = pool;
		m_pending_pools.pop_front();
		EXIT_IF(m_graphics.device.resetDescriptorPool(m_current_pool, {}) != vk::Result::eSuccess);
	} else {
		CreateDescriptorPool();
	}

	m_sets.clear();
	auto& fresh_batch = m_sets[layout];
	if (!Allocate(layout, fresh_batch)) {
		// A pool reused from before a larger layout appeared can be too small for it: a new one,
		// sized for the largest layout seen (RequireDescriptorPoolSize).
		m_pending_pools.emplace_back(m_current_pool, m_master_semaphore.CurrentTick());
		CreateDescriptorPool();
		EXIT_IF(!Allocate(layout, fresh_batch));
	}
	return fresh_batch.sets[--fresh_batch.size];
}

bool DescriptorHeap::Allocate(vk::DescriptorSetLayout layout, Batch& batch) {
	std::array<vk::DescriptorSetLayout, DescriptorSetBatch> layouts;
	layouts.fill(layout);

	vk::DescriptorSetAllocateInfo allocate {};
	allocate.descriptorPool = m_current_pool;
	allocate.pSetLayouts    = layouts.data();

	for (;;) {
		allocate.descriptorSetCount = batch.allocation;
		const auto result = m_graphics.device.allocateDescriptorSets(&allocate, batch.sets.data());
		if (result == vk::Result::eSuccess) {
			batch.size = batch.allocation;
			return true;
		}
		EXIT_IF(result != vk::Result::eErrorOutOfPoolMemory &&
		        result != vk::Result::eErrorFragmentedPool);
		if (batch.allocation == 1) {
			return false;
		}
		batch.allocation /= 2;
	}
}

void DescriptorHeap::CreateDescriptorPool() {
	auto sizes = DescriptorPoolSizes;
	for (size_t index = 0; index < sizes.size(); index++) {
		sizes[index].descriptorCount = std::max(
		    sizes[index].descriptorCount, g_required_pool_sizes[index].load(std::memory_order_relaxed));
	}
	vk::DescriptorPoolCreateInfo create {};
	create.maxSets       = DescriptorHeapCount;
	create.poolSizeCount = static_cast<uint32_t>(sizes.size());
	create.pPoolSizes    = sizes.data();
	EXIT_IF(m_graphics.device.createDescriptorPool(&create, nullptr, &m_current_pool) !=
	        vk::Result::eSuccess);
}

} // namespace Libs::Graphics
