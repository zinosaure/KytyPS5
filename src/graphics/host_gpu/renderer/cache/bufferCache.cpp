#include "graphics/host_gpu/renderer/cache/bufferCache.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "kernel/memory.h"

#include <algorithm>
#include <bit>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <unistd.h>

namespace Libs::Graphics {

namespace {

constexpr uint64_t MiB           = 1024 * 1024;
constexpr uint64_t GdsBufferSize = 64 * 1024;
// GPU-modified ranges come as many small scattered islands, so a readback widens to a window.
constexpr uint64_t DownloadWindowSize = 512 * 1024;

std::pair<Buffer*, uint64_t> UploadMemory(GraphicContext& graphics, CommandScheduler& scheduler,
                                        StreamBuffer& staging, uint64_t size, uint64_t alignment,
                                        vk::BufferUsageFlags usage, const auto& write) {
	const auto [mapped, offset] = staging.Map(size, alignment);
	if (mapped != nullptr) {
		write(mapped, offset);
		staging.Commit();
		return {&staging, offset};
	}

	// Large transfers use the same one-time host allocation as ordinary buffer uploads.
	auto temporary = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Upload, 0,
	                                         usage, size);
	write(temporary->Mapped().data(), 0);
	temporary->Flush(0, size);
	auto* buffer = temporary.get();
	scheduler.DeferOperation([owner = std::move(temporary)]() mutable { owner.reset(); });
	return {buffer, 0};
}

} // namespace

void BufferCache::WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source,
                                  uint64_t size) {
	auto* bytes = static_cast<const uint8_t*>(source);
	while (size != 0) {
		const auto chunk  = std::min(size, m_staging_buffer.Size());
		// Copy waits for ring reuse and flushes the fresh host data before submission.
		const auto offset = m_staging_buffer.Copy(bytes, chunk, 4);
		const auto destination_offset = buffer.Offset(address);
		EXIT_IF(destination_offset > buffer.Size() || chunk > buffer.Size() - destination_offset);
		m_scheduler.EndRendering();
		const auto command = m_scheduler.Current().Handle();
		vk::BufferMemoryBarrier2 before {};
		before.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		before.srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		before.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		before.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer        = buffer.Handle();
		before.offset        = destination_offset;
		before.size          = chunk;
		vk::DependencyInfo dependency {};
		dependency.dependencyFlags          = vk::DependencyFlagBits::eByRegion;
		dependency.bufferMemoryBarrierCount = 1;
		dependency.pBufferMemoryBarriers    = &before;
		command.pipelineBarrier2(dependency);
		const vk::BufferCopy copy {offset, destination_offset, chunk};
		command.copyBuffer(m_staging_buffer.Handle(), buffer.Handle(), 1, &copy);
		auto after          = before;
		after.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		after.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
		after.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		after.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		dependency.pBufferMemoryBarriers = &after;
		command.pipelineBarrier2(dependency);
		bytes += chunk;
		address += chunk;
		size -= chunk;
	}
}

void BufferCache::Register(BufferId id) {
	ChangeRegister<true>(id);
}

void BufferCache::Unregister(BufferId id) {
	ChangeRegister<false>(id);
}

template <bool insert>
void BufferCache::ChangeRegister(BufferId id) {
	auto& buffer = m_slot_buffers[id];
	PageTable::PageRange pages {};
	EXIT_IF(!(GuestRange {buffer.CpuAddress(), buffer.Size()}.Valid()) ||
	        !PageTable::TryGetPageRange(buffer.CpuAddress(), buffer.Size(), pages));
	for (size_t page = pages.first; page < pages.last_exclusive; ++page) {
		if constexpr (insert) {
			m_page_table[page] = id;
		} else {
			m_page_table[page] = {};
		}
	}
	const auto size_pages = pages.last_exclusive - pages.first;
	const auto table_offset = PageIndex(buffer.CpuAddress()) * sizeof(vk::DeviceAddress);
	if constexpr (insert) {
		m_buffers.emplace(buffer.CpuAddress(), id);
		m_total_used_memory += buffer.Size();
		buffer.lru_id = m_lru_cache.Insert(id, m_gc_tick);
		std::vector<vk::DeviceAddress> addresses;
		addresses.reserve(size_pages);
		for (uint64_t i = 0; i < size_pages; ++i) {
			addresses.push_back(buffer.BufferDeviceAddress() + (i << CACHING_PAGEBITS));
		}
		WriteDataBuffer(m_bda_pagetable_buffer, table_offset,
		                addresses.data(), addresses.size() * sizeof(vk::DeviceAddress));
		QueueBdaSync(buffer.CpuAddress(), buffer.Size());
	} else {
		m_buffers.erase(m_buffers.find(buffer.CpuAddress()));
		m_total_used_memory -= buffer.Size();
		m_lru_cache.Free(buffer.lru_id);
		m_bda_pagetable_buffer.Fill(table_offset,
		                            size_pages * sizeof(vk::DeviceAddress), 0);
		buffer.is_deleted = true;
	}
}

void BufferCache::TouchBuffer(const Buffer& buffer) {
	m_lru_cache.Touch(buffer.lru_id, m_gc_tick);
}

void BufferCache::DeleteBuffer(BufferId id) {
	Unregister(id);
	if (m_scheduler.Active()) {
		m_scheduler.DeferOperation([this, id] { m_slot_buffers.erase(id); });
	} else {
		m_slot_buffers.erase(id);
	}
}

template <bool async>
bool BufferCache::DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size     = 0;
	const auto                  buffer_address = buffer.CpuAddress();
	m_memory_tracker.ForEachDownloadRange<false>(
	    vaddr, size, [&](uint64_t address, uint64_t bytes) noexcept {
		    m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, address, bytes,
		                                           "buffer download");
		    m_gpu_modified_ranges.ForEachInRange(address, bytes, [&](uint64_t start, uint64_t end) {
			    copies.emplace_back(start - buffer_address, total_size, end - start);
			    // Start each packed range on its own 64-byte cache line.
			    total_size += Common::AlignUp(end - start, 64);
		    });
		    m_gpu_modified_ranges.Subtract(address, bytes);
		    ForgetColorMetadataChecks(address, bytes);
	    });
	if (copies.empty()) {
		return false;
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferDownloadBytes, total_size);

	auto [mapped, offset] = m_download_buffer.Map(total_size, 64);
	std::unique_ptr<Buffer> temporary;
	if (mapped == nullptr) {
		temporary = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                     vk::BufferUsageFlagBits::eTransferDst, total_size);
		mapped = temporary->Mapped().data();
	} else {
		m_download_buffer.Commit();
	}
	const auto& download = temporary ? *temporary : m_download_buffer;
	for (auto& copy: copies) {
		copy.dstOffset += offset;
	}

	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto              native = command.Handle();
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = buffer.Handle();
	before.offset              = 0;
	before.size                = buffer.Size();
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &before, 0,
	                       nullptr);
	native.copyBuffer(buffer.Handle(), download.Handle(),
	                  static_cast<uint32_t>(copies.size()), copies.data());

	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	after.buffer        = download.Handle();
	after.offset        = offset;
	after.size          = total_size;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands |
	                           vk::PipelineStageFlagBits::eHost,
	                       {}, 0, nullptr, 1, &after, 0, nullptr);
	auto publish = [this, mapped, offset, total_size, buffer_address,
	                copies = std::move(copies), owner = std::move(temporary)] {
		(owner ? *owner : m_download_buffer).Invalidate(offset, total_size);
		for (const auto& copy: copies) {
			Libs::LibKernel::Memory::WriteBacking(buffer_address + copy.srcOffset,
			                                      mapped + (copy.dstOffset - offset), copy.size);
		}
	};
	if constexpr (async) {
		m_scheduler.DeferPriorityOperation(std::move(publish));
	} else {
		const auto tick = m_scheduler.CurrentTick();
		m_scheduler.Wait(tick);
		m_scheduler.WaitPriorityOperations(tick);
		publish();
	}
	return true;
}

BufferCache::BufferCache(GraphicContext& graphics, CommandScheduler& scheduler,
                         PageManager& page_manager, TextureCache& texture_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_fault_manager(graphics, scheduler, *this),
      m_gds_buffer(graphics, scheduler, MemoryUsage::Stream, 0, AllFlags, GdsBufferSize),
      m_bda_pagetable_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
                             BDA_PAGETABLE_SIZE),
      m_memory_tracker(page_manager),
      m_staging_buffer(graphics, scheduler, MemoryUsage::Upload, 512 * MiB),
      m_stream_buffer(graphics, scheduler, MemoryUsage::Stream, 64 * MiB),
      m_download_buffer(graphics, scheduler, MemoryUsage::Download, 64 * MiB),
      m_device_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 128 * MiB),
      m_texture_cache(texture_cache) {
	std::memset(m_gds_buffer.Mapped().data(), 0, static_cast<size_t>(m_gds_buffer.Size()));
	m_gds_buffer.Flush(0, m_gds_buffer.Size());
	SetVulkanObjectNameF(m_graphics.device, m_bda_pagetable_buffer.Handle(),
	                     "BDA Page Table Buffer");
	m_scheduler.BeginCommand();
	for (auto* buffer: {&m_bda_pagetable_buffer, m_fault_manager.GetFaultBuffer()}) {
		buffer->Fill(0, buffer->Size(), 0);
	}
	const auto null_id =
	    m_slot_buffers.insert(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0, AllFlags, 16);
	EXIT_IF(null_id != NULL_BUFFER_ID);
	SetVulkanObjectNameF(m_graphics.device, GetBuffer(null_id).Handle(), "Kyty.NullBuffer");
	if (m_graphics.unified_memory_import) {
		ImportUnifiedMemory();
	}
	if (!m_graphics.CanReportMemoryUsage()) {
		return;
	}
	constexpr int64_t GiB              = 1024ll * 1024 * 1024;
	constexpr int64_t target_threshold = 8 * GiB;
	const auto        budget =
	    static_cast<int64_t>(std::min<uint64_t>(m_graphics.GetTotalMemoryBudget(), INT64_MAX));
	const auto threshold = std::min(budget, target_threshold);
	const auto expected  = std::min(budget - 6 * threshold / 10, budget - GiB);
	const auto critical  = std::min(budget - 2 * threshold / 10, budget - GiB / 2);
	m_trigger_gc_memory  = static_cast<uint64_t>(std::max<int64_t>(expected, GiB));
	m_critical_gc_memory = static_cast<uint64_t>(std::max<int64_t>(critical, 2 * GiB));
}

BufferCache::~BufferCache() {
	if (m_unified_memory) {
		// Views of the guest's memory are destroyed with the slots, after it: Vulkan allows it.
		if (m_unified_bind_fence) {
			m_graphics.device.destroyFence(m_unified_bind_fence);
		}
		m_graphics.device.freeMemory(m_unified_memory);
		m_unified_memory = nullptr;
	}
	if (!m_gpu_modified_ranges.Empty()) {
		EXIT("BufferCache: destroyed with pending GPU-modified ranges\n");
	}
	for (const auto& [vaddr, id]: m_buffers) {
		(void)vaddr;
		const auto& buffer = m_slot_buffers[id];
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: destroyed with GPU-modified buffer\n");
		}
	}
}

void BufferCache::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	m_memory_tracker.InvalidateRegion(vaddr, size,
	                                  [this, vaddr, size] { ReadMemory(vaddr, size, true); });
	QueueBdaSync(vaddr, size);
}

void BufferCache::ReadMemory(uint64_t vaddr, uint64_t size, bool is_write) {
	if (!GuestGpu::IsGpuThread() && CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported buffer readback from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	m_scheduler.Context().GetGpu().SendCommandSync([this, vaddr, size, is_write] {
		// Unified memory: nothing to bring back, the bytes are in the guest memory once the GPU
		// work that writes them is done. A copy back would also land after newer GPU writes.
		if (IsUnified()) {
			FinishPendingGpuWrites(vaddr, size);
			return;
		}
		if (is_write && !IsRegionRegistered(vaddr, size)) {
			return;
		}
		auto& buffer = m_slot_buffers[FindBuffer(vaddr, size)];
		if (IsRegionGpuModified(vaddr, size)) {
			const auto start =
			    std::max(Common::AlignDown(vaddr, DownloadWindowSize), buffer.CpuAddress());
			const auto end = std::min(std::max(start + DownloadWindowSize, vaddr + size),
			                          buffer.CpuAddress() + buffer.Size());
			if (DownloadBufferMemory<false>(buffer, start, end - start)) {
				m_memory_tracker.UnmarkRegionAsGpuModified(start, end - start);
			}
		}
		if (is_write) {
			m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
			QueueBdaSync(vaddr, size);
		}
	});
}

BufferId BufferCache::FindBuffer(uint64_t vaddr, uint64_t size) {
	if (vaddr == 0) {
		return NULL_BUFFER_ID;
	}
	if (IsUnified()) {
		if (const auto window = UnifiedWindow(vaddr, size)) {
			return window;
		}
	}
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid buffer discovery request\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			return *owner;
		}
	}
	return CreateBuffer(vaddr, size);
}

BufferCache::OverlapResult BufferCache::ResolveOverlaps(uint64_t vaddr, uint64_t size) {
	static constexpr int      StreamLeapThreshold = 16;
	static constexpr uint64_t StreamLeapSize      = CACHING_PAGESIZE * 128;

	auto       begin      = vaddr;
	auto       end        = vaddr + size;
	const auto find_first = [&](uint64_t address) {
		auto first = m_buffers.lower_bound(address);
		if (first != m_buffers.begin()) {
			const auto  previous = std::prev(first);
			const auto& buffer   = m_slot_buffers[previous->second];
			if (buffer.CpuAddress() + buffer.Size() > address) {
				first = previous;
			}
		}
		return first;
	};
	auto first           = find_first(begin);
	auto last            = first;
	int  stream_score    = 0;
	bool has_stream_leap = false;
	for (; last != m_buffers.end() && last->first < end; ++last) {
		const auto& buffer        = m_slot_buffers[last->second];
		const auto  buffer_begin  = buffer.CpuAddress();
		const auto  buffer_end    = buffer_begin + buffer.Size();
		const bool  expands_left  = buffer_begin < begin;
		const bool  expands_right = buffer_end > end;
		begin                     = std::min(begin, buffer_begin);
		end                       = std::max(end, buffer_end);
		if (!has_stream_leap && (stream_score += buffer.StreamScore()) > StreamLeapThreshold) {
			has_stream_leap = true;
			// Reserve space in the incoming stream's direction of growth.
			// The old buffer extending left of the request predicts growth to the right, and vice versa.
			if (expands_left) {
				end += std::min(StreamLeapSize, (vaddr < LOWER_ADDRESS_SIZE ? LOWER_ADDRESS_SIZE
				                                       : LibKernel::Memory::kExtendedMemoryBase +
				                                             LibKernel::Memory::kExtendedMemorySize) - end);
			}
			if (expands_right) {
				const auto minimum = vaddr < LOWER_ADDRESS_SIZE
				                         ? CACHING_PAGESIZE * 2
				                         : LibKernel::Memory::kExtendedMemoryBase;
				if (begin > minimum) {
					begin -= std::min(StreamLeapSize, begin - minimum);
				}
				first = find_first(begin);
				begin = std::min(begin, first->first);
			}
		}
	}
	return {first, last, begin, end, has_stream_leap};
}

void BufferCache::JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score) {
	auto& new_buffer = m_slot_buffers[new_id];
	auto& overlap    = m_slot_buffers[overlap_id];
	if (accumulate_stream_score) {
		new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
	}
	new_buffer.CopyFrom(m_scheduler.Current(), overlap, 0,
	                    overlap.CpuAddress() - new_buffer.CpuAddress(), overlap.Size());
	DeleteBuffer(overlap_id);
}

BufferId BufferCache::CreateBuffer(uint64_t vaddr, uint64_t size) {
	const auto end = Common::AlignUp(vaddr + size, CACHING_PAGESIZE);
	vaddr = Common::AlignDown(vaddr, CACHING_PAGESIZE);
	size               = end - vaddr;
	const auto overlap = ResolveOverlaps(vaddr, size);

	const auto id = m_slot_buffers.insert(
	    m_graphics, m_scheduler, MemoryUsage::DeviceLocal, overlap.begin,
	    AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, overlap.end - overlap.begin);
	const auto& buffer = m_slot_buffers[id];
	SetVulkanObjectNameF(m_graphics.device, buffer.Handle(),
	                     "Kyty.GameBuffer[guest=0x{:016x} size=0x{:x}]", overlap.begin,
	                     overlap.end - overlap.begin);
	for (auto it = overlap.first; it != overlap.last;) {
		const auto old_id = (it++)->second;
		JoinOverlap(id, old_id, !overlap.has_stream_leap);
	}
	Register(id);
	return id;
}

bool BufferCache::SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size, bool is_written,
                                    bool is_texel_buffer) {
	if (buffer.IsExternal()) {
		return is_texel_buffer && !is_written && SynchronizeBufferFromImage(buffer, vaddr, size);
	}
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size = 0;
	vk::Buffer                  source;
	m_memory_tracker.ForEachUploadRange(
	    vaddr, size, is_written,
	    [&](uint64_t address, uint64_t bytes) noexcept {
		    copies.emplace_back(total_size, buffer.Offset(address), bytes);
		    total_size += bytes;
	    },
	    [&]() noexcept { source = UploadCopies(buffer, copies, total_size); });
	if (source) {
		auto& command = m_scheduler.Current();
		command.EndRendering();
		const auto native = command.Handle();
		vk::BufferMemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
		                       vk::AccessFlagBits::eTransferRead |
		                       vk::AccessFlagBits::eTransferWrite;
		before.dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer              = buffer.Handle();
		before.offset              = 0;
		before.size                = buffer.Size();
		native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                       vk::PipelineStageFlagBits::eTransfer,
		                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &before, 0, nullptr);
		native.copyBuffer(source, buffer.Handle(), static_cast<uint32_t>(copies.size()),
		                  copies.data());
		auto after          = before;
		after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
		native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                       vk::PipelineStageFlagBits::eAllCommands,
		                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &after, 0, nullptr);
	}
	if (is_texel_buffer && !is_written) {
		return SynchronizeBufferFromImage(buffer, vaddr, size);
	}
	return false;
}

vk::Buffer BufferCache::UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                                     uint64_t total_size) {
	if (copies.empty()) {
		return nullptr;
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferUploadBytes, total_size);

	const auto source = UploadMemory(m_graphics, m_scheduler, m_staging_buffer, total_size, 4,
	    vk::BufferUsageFlagBits::eTransferSrc, [&](uint8_t* mapped, uint64_t base_offset) {
		for (auto& copy: copies) {
			const auto address = buffer.CpuAddress() + copy.dstOffset;
			if (!LibKernel::Memory::CopySparseMemory(address, mapped + copy.srcOffset, copy.size)) {
				EXIT("BufferCache: failed to read sparse guest buffer memory\n");
			}
			copy.srcOffset += base_offset;
		}
	});
	return source.first->Handle();
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBuffer(uint64_t vaddr, uint64_t size,
                                                       bool is_written, bool is_texel_buffer,
                                                       BufferId id) {
	EXIT_IF(!(GuestRange {vaddr, size}.Valid()));

	if (IsUnified()) {
		const auto unified = UnifiedWindow(vaddr, size);
		if (unified) {
			// The windows live as long as the cache: they have no LRU entry to touch.
			auto& buffer = m_slot_buffers[unified];
			// An image newer than the memory under a texel read still lands in it first.
			if (is_texel_buffer && !is_written) {
				(void)SynchronizeBufferFromImage(buffer, vaddr, size);
			}
			if (is_written) {
				ForgetColorMetadataChecks(vaddr, size);
				RecordUnifiedGpuWrite(vaddr, size);
			}
			return {&buffer, buffer.Offset(vaddr)};
		}
	}

	if (!is_written && size <= CACHING_PAGESIZE &&
	    !m_memory_tracker.IsRegionGpuModified(vaddr, size) &&
	    m_memory_tracker.IsRegionCpuModified(vaddr, size)) {
		const auto alignment = std::max<uint64_t>(
		    m_graphics.physical_device_properties.limits.minUniformBufferOffsetAlignment, 1);
		auto [mapped, offset] = m_stream_buffer.Map(size, alignment, false);
		if (mapped != nullptr) {
			if (!LibKernel::Memory::CopySparseMemory(vaddr, mapped, size)) {
				EXIT("BufferCache: failed to read sparse guest buffer memory\n");
			}
			m_stream_buffer.Commit();
			return {&m_stream_buffer, offset};
		}
	}

	if (IsBufferInvalid(id) || !m_slot_buffers[id].IsInBounds(vaddr, size)) {
		id = FindBuffer(vaddr, size);
	}
	auto& buffer = m_slot_buffers[id];
	TouchBuffer(buffer);
	(void)SynchronizeBuffer(buffer, vaddr, size, is_written, is_texel_buffer);
	if (is_written) {
		m_gpu_modified_ranges.Add(vaddr, size);
		ForgetColorMetadataChecks(vaddr, size);
	}
	return {&buffer, buffer.Offset(vaddr)};
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBufferForImage(uint64_t vaddr, uint64_t size) {
	// Unified memory: the GPU reads the image from the guest memory, after the work recorded before
	// it, which may still write there (a DMA copy, a shader). A copy made now would miss it.
	if (IsUnified()) {
		if (const auto unified = UnifiedWindow(vaddr, size)) {
			auto& buffer = m_slot_buffers[unified];
			return {&buffer, buffer.Offset(vaddr)};
		}
		FinishPendingGpuWrites(vaddr, size);
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			TouchBuffer(buffer);
			(void)SynchronizeBuffer(buffer, vaddr, size, false, false);
			return {&buffer, buffer.Offset(vaddr)};
		}
	}
	if (IsRegionGpuModified(vaddr, size)) {
		return ObtainBuffer(vaddr, size, false, false);
	}

	return UploadMemory(m_graphics, m_scheduler, m_staging_buffer, size, 16,
	    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eStorageBuffer,
	    [&](uint8_t* mapped, uint64_t) {
		if (!Libs::LibKernel::Memory::TryReadSparseBacking(vaddr, mapped, size)) {
			EXIT("BufferCache: failed to read mapped guest image backing\n");
		}
	});
}

void BufferCache::FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds) {
	if ((vaddr & 3u) != 0 || size == 0 || (size & 3u) != 0 || size > UINT64_MAX - vaddr) {
		EXIT("BufferCache: fill range must be dword aligned\n");
	}
	if (is_gds) {
		if (vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - vaddr) {
			EXIT("BufferCache: GDS fill range is out of bounds\n");
		}
		m_gds_buffer.Fill(vaddr, size, value);
		return;
	}
	if (vaddr == 0) {
		EXIT("BufferCache: invalid fill memory address\n");
	}
	(void)m_texture_cache.ClearMeta(vaddr);
	if (!IsUnified() && !IsRegionGpuModified(vaddr, size)) {
		// Access the guest mapping so write faults invalidate cached buffers and images.
		auto* destination = reinterpret_cast<uint32_t*>(vaddr);
		std::fill(destination, destination + size / sizeof(uint32_t), value);
		return;
	}

	m_texture_cache.InvalidateMemoryFromGPU(vaddr, size);
	if (IsUnified()) {
		// The CPU fill this replaces faulted on every image over the range, not just the one that
		// starts there: a mip level, a slice. They reload from the guest memory, after the fill.
		m_texture_cache.InvalidateMemory(vaddr, size);
	}
	auto [dst, dst_offset] = ObtainBuffer(vaddr, size, true, true);
	dst->Fill(dst_offset, size, value);
}

void BufferCache::CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
                             bool src_gds) {
	const bool dst_memory = !dst_gds;
	const bool src_memory = !src_gds;
	if ((dst_memory && dst_vaddr == 0) || (src_memory && src_vaddr == 0) || size == 0 ||
	    ((dst_gds || src_gds) && ((dst_vaddr | src_vaddr | size) & 3u) != 0) ||
	    size > UINT64_MAX - dst_vaddr || size > UINT64_MAX - src_vaddr || (dst_gds && src_gds) ||
	    (dst_gds && (dst_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - dst_vaddr)) ||
	    (src_gds && (src_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - src_vaddr))) {
		EXIT("BufferCache: invalid copy range, src=0x%016" PRIx64 " dst=0x%016" PRIx64
		     " size=0x%016" PRIx64 " src_gds=%d dst_gds=%d\n",
		     src_vaddr, dst_vaddr, size, static_cast<int>(src_gds), static_cast<int>(dst_gds));
	}
	if (!IsUnified() && src_memory && dst_memory && !IsRegionGpuModified(dst_vaddr, size) &&
	    !IsRegionGpuModified(src_vaddr, size) && !m_texture_cache.FindImageFromRange(src_vaddr, size)) {
		std::memcpy(reinterpret_cast<void*>(dst_vaddr), reinterpret_cast<const void*>(src_vaddr),
		            size);
		return;
	}

	auto& command = m_scheduler.Current();
	if (dst_memory) {
		m_texture_cache.InvalidateMemoryFromGPU(dst_vaddr, size);
		if (IsUnified()) {
			// As in FillBuffer: every image the copy writes into, after it.
			m_texture_cache.InvalidateMemory(dst_vaddr, size);
		}
	}
	const auto src_id      = src_memory ? FindBuffer(src_vaddr, size) : BufferId {};
	const auto dst_id      = dst_memory ? FindBuffer(dst_vaddr, size) : BufferId {};
	auto [src, src_offset] = src_memory ? ObtainBuffer(src_vaddr, size, false, true, src_id)
	                                    : std::pair {&m_gds_buffer, src_vaddr};
	auto [dst, dst_offset] = dst_memory ? ObtainBuffer(dst_vaddr, size, true, true, dst_id)
	                                    : std::pair {&m_gds_buffer, dst_vaddr};
	dst->CopyFrom(command, *src, src_offset, dst_offset, size);
}

bool BufferCache::IsRegionRegistered(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid registered-region query\n");
	}
	if (IsUnified()) {
		auto view = m_unified_views.upper_bound(vaddr + size - 1);
		if (view != m_unified_views.begin() && std::prev(view)->first + std::prev(view)->second.first > vaddr) {
			return true;
		}
	}
	// Cached buffers are ordered and non-overlapping. The last buffer beginning before the query
	// end is therefore the only possible intersection.
	const auto candidate = m_buffers.lower_bound(vaddr + size);
	if (candidate == m_buffers.begin()) {
		return false;
	}
	const auto& [address, id] = *std::prev(candidate);
	return address + m_slot_buffers[id].Size() > vaddr;
}

void BufferCache::MarkColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys) {
	for (uint32_t slot = 0; slot < m_checked_color_metadata.size(); slot++) {
		if ((keys & (1u << slot)) != 0) {
			m_checked_color_metadata[slot].Add(vaddr, size);
		}
	}
}

bool BufferCache::IsColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys) const {
	for (uint32_t slot = 0; slot < m_checked_color_metadata.size(); slot++) {
		if ((keys & (1u << slot)) != 0 && !m_checked_color_metadata[slot].Contains(vaddr, size)) {
			return false;
		}
	}
	return true;
}

void BufferCache::ForgetColorMetadataChecks(uint64_t vaddr, uint64_t size) {
	for (auto& checked: m_checked_color_metadata) {
		checked.Subtract(vaddr, size);
	}
}

bool BufferCache::IsRegionGpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionGpuModified(vaddr, size);
}

bool BufferCache::HasGpuDirtyBytes(uint64_t vaddr, uint64_t size) {
	// Unified memory has no downloads: the bytes are in the guest memory once the GPU is done.
	return m_gpu_modified_ranges.Intersects(vaddr, size) || HasPendingGpuWrites(vaddr, size);
}

uint64_t BufferCache::GpuDirtyBytes(uint64_t vaddr, uint64_t size) {
	uint64_t bytes = 0;
	m_gpu_modified_ranges.ForEachInRange(vaddr, size,
	                                     [&](uint64_t begin, uint64_t end) { bytes += end - begin; });
	if (HasPendingGpuWrites(vaddr, size)) {
		bytes += m_pending_gpu_writes.Bytes(vaddr, size);
	}
	return std::min(bytes, size);
}

bool BufferCache::IsRegionCpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionCpuModified(vaddr, size);
}

void BufferCache::RunGarbageCollector() {
	const auto tick = m_gc_tick++;
	if (m_graphics.CanReportMemoryUsage()) {
		m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
	}
	if (m_total_used_memory < m_trigger_gc_memory) {
		return;
	}

	const bool     aggressive = m_total_used_memory >= m_critical_gc_memory;
	const uint64_t age        = std::min<uint64_t>(aggressive ? 80 : 160, tick);
	const size_t   limit      = aggressive ? 64 : 32;

	std::vector<BufferId> dirty_buffers;
	size_t                retire_count = 0;
	m_lru_cache.ForEachItemBelow(tick - age, [&](BufferId id) {
		auto& buffer = m_slot_buffers[id];
		m_memory_tracker.ValidateGpuDirtyOwnership(m_gpu_modified_ranges, buffer.CpuAddress(),
		                                           buffer.Size(), "garbage collection");
		const bool dirty = m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size());
		if (dirty && !aggressive) {
			return false;
		}
		if (dirty) {
			EXIT_NOT_IMPLEMENTED(!DownloadBufferMemory<true>(buffer, buffer.CpuAddress(), buffer.Size()));
			dirty_buffers.push_back(id);
		} else {
			m_memory_tracker.MarkRegionAsCpuModified(buffer.CpuAddress(), buffer.Size());
			DeleteBuffer(id);
			Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferEvictions);
		}
		return ++retire_count == limit;
	});
	if (dirty_buffers.empty()) {
		return;
	}

	// Publish all queued downloads before releasing their tracked pages and owners.
	const auto completion_tick = m_scheduler.CurrentTick();
	m_scheduler.Wait(completion_tick);
	m_scheduler.WaitPriorityOperations(completion_tick);
	for (const auto id: dirty_buffers) {
		auto& buffer = m_slot_buffers[id];
		m_memory_tracker.UnmarkRegionAsGpuModified(buffer.CpuAddress(), buffer.Size());
		m_memory_tracker.MarkRegionAsCpuModified(buffer.CpuAddress(), buffer.Size());
		Unregister(id);
		m_slot_buffers.erase(id);
	}
}

void BufferCache::ProcessFaultBuffer() {
	m_fault_manager.ProcessFaultBuffer();
}

void BufferCache::QueueBdaSync(uint64_t vaddr, uint64_t size) {
	const auto begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto end   = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	std::lock_guard lock(m_bda_sync_mutex);
	m_bda_sync.Add(begin, end - begin);
}

RangeSet BufferCache::TakeBdaSync() {
	std::lock_guard lock(m_bda_sync_mutex);
	return std::exchange(m_bda_sync, {});
}

void BufferCache::SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size) {
	if (IsUnified()) {
		(void)BindUnified(vaddr, size);
	}
	const auto end = vaddr + size;
	auto       it  = m_buffers.upper_bound(vaddr);
	if (it != m_buffers.begin()) {
		--it;
	}
	for (; it != m_buffers.end() && it->first < end; ++it) {
		auto&      buffer = m_slot_buffers[it->second];
		const auto start  = std::max(buffer.CpuAddress(), vaddr);
		const auto finish = std::min(buffer.CpuAddress() + buffer.Size(), end);
		if (start < finish) {
			(void)SynchronizeBuffer(buffer, start, finish - start, false, false);
		}
	}
}

void BufferCache::ImportUnifiedMemory() {
	const int fd = dup(LibKernel::Memory::UnifiedMemoryFd());
	EXIT_IF(fd < 0);
	vk::MemoryFdPropertiesKHR properties {};
	if (m_graphics.device.getMemoryFdPropertiesKHR(
	        vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT, fd, &properties) !=
	        vk::Result::eSuccess ||
	    properties.memoryTypeBits == 0) {
		close(fd);
		Log::WriteToConsoleAndLog("GPU: unified memory off, the device cannot import it.\n");
		return;
	}
	const auto type = static_cast<uint32_t>(std::countr_zero(properties.memoryTypeBits));
	vk::ImportMemoryFdInfoKHR import {};
	import.handleType = vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT;
	import.fd         = fd;
	vk::MemoryAllocateFlagsInfo flags {};
	flags.pNext = &import;
	flags.flags = vk::MemoryAllocateFlagBits::eDeviceAddress;
	vk::MemoryAllocateInfo allocation {};
	allocation.pNext           = &flags;
	allocation.allocationSize  = LibKernel::Memory::UnifiedMemorySize();
	allocation.memoryTypeIndex = type;
	if (m_graphics.device.allocateMemory(&allocation, nullptr, &m_unified_memory) !=
	    vk::Result::eSuccess) {
		close(fd);
		m_unified_memory = nullptr;
		Log::WriteToConsoleAndLog("GPU: unified memory off, the import failed.\n");
		return;
	}
	m_unified_base                  = LibKernel::Memory::UnifiedMemoryBase();
	m_graphics.unified_memory_bytes = allocation.allocationSize;
	Log::WriteToConsoleAndLog(fmt::format("GPU: unified guest memory, {} MiB imported (type {}).\n",
	                                      allocation.allocationSize >> 20u, type));
}

BufferId BufferCache::UnifiedWindow(uint64_t vaddr, uint64_t size) {
	const auto index = vaddr >> UnifiedWindowBits;
	const auto base  = index << UnifiedWindowBits;
	if (size == 0 || vaddr + size > base + UnifiedWindowSize) {
		return {};
	}
	// Bound already? Every page of the range must be one of the guest's views.
	const auto page_begin = Common::AlignDown(vaddr, CACHING_PAGESIZE);
	const auto page_end   = Common::AlignUp(vaddr + size, CACHING_PAGESIZE);
	bool       bound      = false;
	{
		auto it = m_unified_views.upper_bound(page_begin);
		if (it != m_unified_views.begin()) {
			--it;
		}
		auto covered = page_begin;
		for (; it != m_unified_views.end() && it->first <= covered && covered < page_end; ++it) {
			covered = std::max(covered, it->first + it->second.first);
		}
		bound = covered >= page_end;
	}
	if (!bound && !BindUnified(page_begin, page_end - page_begin)) {
		return {};
	}
	auto& id = m_unified_windows[index];
	if (!id) {
		id = m_slot_buffers.insert(m_graphics, m_scheduler, base,
		                           AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress,
		                           UnifiedWindowSize, SparseWindow {});
		SetVulkanObjectNameF(m_graphics.device, m_slot_buffers[id].Handle(),
		                     "Kyty.GuestWindow[guest=0x{:016x}]", base);
	}
	return id;
}

bool BufferCache::BindUnified(uint64_t vaddr, uint64_t size) {
	const auto begin = Common::AlignDown(vaddr, CACHING_PAGESIZE);
	const auto end   = Common::AlignUp(vaddr + size, CACHING_PAGESIZE);
	std::vector<std::tuple<uint64_t, uint64_t, uint64_t>> pieces;
	uint64_t                                              backed = 0;
	LibKernel::Memory::ForEachUnifiedView(begin, end - begin,
	                                      [&](uint64_t address, uint64_t bytes, uint64_t offset) {
		                                      pieces.emplace_back(address, bytes, offset);
		                                      backed += bytes;
	                                      });
	// Unchanged since the last bind: nothing to do.
	std::vector<std::tuple<uint64_t, uint64_t, uint64_t>> bound;
	{
		auto it = m_unified_views.upper_bound(begin);
		if (it != m_unified_views.begin() && std::prev(it)->first + std::prev(it)->second.first > begin) {
			--it;
		}
		for (; it != m_unified_views.end() && it->first < end; ++it) {
			const auto piece_begin = std::max(begin, it->first);
			const auto piece_end   = std::min(end, it->first + it->second.first);
			bound.emplace_back(piece_begin, piece_end - piece_begin,
			                   it->second.second + (piece_begin - it->first));
		}
	}
	if (pieces != bound) {
		BindUnifiedPieces(begin, end, pieces);
	}
	return backed == end - begin;
}

void BufferCache::BindUnifiedPieces(uint64_t begin, uint64_t end,
                                    const std::vector<std::tuple<uint64_t, uint64_t, uint64_t>>& pieces) {
	// The GPU may still read the old views: let it finish before they change. A range the game
	// has just mapped has none, and the GPU has nothing there to finish.
	auto bound = m_unified_views.upper_bound(begin);
	if (bound != m_unified_views.begin() && std::prev(bound)->first + std::prev(bound)->second.first > begin) {
		--bound;
	}
	if (bound != m_unified_views.end() && bound->first < end && m_scheduler.Active()) {
		m_scheduler.Finish();
	}
	// Every window that covers part of [begin, end): the one of its GiB and the one before.
	std::vector<vk::SparseMemoryBind>               binds;
	std::vector<std::pair<Buffer*, uint32_t>>       windows; // buffer, first bind
	const auto first_index = (begin >> UnifiedWindowBits) == 0 ? 0 : (begin >> UnifiedWindowBits) - 1;
	for (auto index = first_index; index <= ((end - 1) >> UnifiedWindowBits); index++) {
		const auto base = index << UnifiedWindowBits;
		const auto stop = base + UnifiedWindowSize;
		const auto from = std::max(begin, base);
		const auto to   = std::min(end, stop);
		if (from >= to) {
			continue;
		}
		auto& id = m_unified_windows[index];
		if (!id) {
			id = m_slot_buffers.insert(m_graphics, m_scheduler, base,
			                           AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress,
			                           UnifiedWindowSize, SparseWindow {});
			SetVulkanObjectNameF(m_graphics.device, m_slot_buffers[id].Handle(),
			                     "Kyty.GuestWindow[guest=0x{:016x}]", base);
		}
		const auto first = static_cast<uint32_t>(binds.size());
		// Unbind the whole part first, then bind the views: later binds win.
		binds.push_back({from - base, to - from, nullptr, 0, {}});
		for (const auto& [address, bytes, offset]: pieces) {
			const auto piece_from = std::max(from, address);
			const auto piece_to   = std::min(to, address + bytes);
			if (piece_from < piece_to) {
				binds.push_back({piece_from - base, piece_to - piece_from, m_unified_memory,
				                 offset + (piece_from - address), {}});
			}
		}
		windows.emplace_back(&m_slot_buffers[id], first);
	}
	std::vector<vk::SparseBufferMemoryBindInfo> buffer_binds;
	for (size_t i = 0; i < windows.size(); i++) {
		const auto first = windows[i].second;
		const auto last  = i + 1 < windows.size() ? windows[i + 1].second : static_cast<uint32_t>(binds.size());
		buffer_binds.push_back({windows[i].first->Handle(), last - first, binds.data() + first});
	}
	vk::BindSparseInfo info {};
	info.bufferBindCount = static_cast<uint32_t>(buffer_binds.size());
	info.pBufferBinds    = buffer_binds.data();
	if (!m_unified_bind_fence) {
		vk::FenceCreateInfo fence_info {};
		EXIT_NOT_IMPLEMENTED(m_graphics.device.createFence(&fence_info, nullptr, &m_unified_bind_fence) !=
		                     vk::Result::eSuccess);
	}
	{
		Common::LockGuard lock(m_graphics.queue_mutex);
		EXIT_NOT_IMPLEMENTED(m_graphics.queue.bindSparse(1, &info, m_unified_bind_fence) !=
		                     vk::Result::eSuccess);
	}
	EXIT_NOT_IMPLEMENTED(m_graphics.device.waitForFences(1, &m_unified_bind_fence, VK_TRUE, UINT64_MAX) !=
	                     vk::Result::eSuccess);
	EXIT_NOT_IMPLEMENTED(m_graphics.device.resetFences(1, &m_unified_bind_fence) != vk::Result::eSuccess);

	// The record of what is bound.
	{
		auto it = m_unified_views.upper_bound(begin);
		if (it != m_unified_views.begin() && std::prev(it)->first + std::prev(it)->second.first > begin) {
			--it;
		}
		std::vector<std::pair<uint64_t, std::pair<uint64_t, uint64_t>>> kept;
		while (it != m_unified_views.end() && it->first < end) {
			const auto view_begin = it->first;
			const auto view_end   = it->first + it->second.first;
			const auto offset     = it->second.second;
			if (view_begin < begin) {
				kept.push_back({view_begin, {begin - view_begin, offset}});
			}
			if (view_end > end) {
				kept.push_back({end, {view_end - end, offset + (end - view_begin)}});
			}
			it = m_unified_views.erase(it);
		}
		for (const auto& entry: kept) {
			m_unified_views.insert(entry);
		}
		for (const auto& [address, bytes, offset]: pieces) {
			m_unified_views[address] = {bytes, offset};
		}
	}

	// The BDA table: each page through the window of its GiB, or nothing.
	std::vector<vk::DeviceAddress> addresses((end - begin) >> CACHING_PAGEBITS, 0);
	for (const auto& [address, bytes, offset]: pieces) {
		for (auto page = address; page < address + bytes; page += CACHING_PAGESIZE) {
			const auto& window = m_slot_buffers[m_unified_windows[page >> UnifiedWindowBits]];
			addresses[(page - begin) >> CACHING_PAGEBITS] =
			    window.BufferDeviceAddress() + (page - window.CpuAddress());
		}
	}
	if (!addresses.empty() && m_scheduler.Active() && !m_scheduler.Current().IsInvalid()) {
		WriteDataBuffer(m_bda_pagetable_buffer, PageIndex(begin) * sizeof(vk::DeviceAddress),
		                addresses.data(), addresses.size() * sizeof(vk::DeviceAddress));
	}
}

bool BufferCache::PendingGuestWrite(uint64_t vaddr, uint64_t size, uint64_t* value) {
	if (m_pending_guest_writes.empty()) {
		return false;
	}
	const auto done = m_scheduler.GetMasterSemaphore().KnownGpuTick();
	std::erase_if(m_pending_guest_writes, [done](const auto& entry) { return entry.second.tick <= done; });
	const auto found = m_pending_guest_writes.find(vaddr);
	if (found == m_pending_guest_writes.end() || found->second.size != size) {
		return false;
	}
	*value = found->second.value;
	return true;
}

void BufferCache::RecordUnifiedGpuWrite(uint64_t vaddr, uint64_t size) {
	m_pending_gpu_writes.Add(vaddr, size, m_scheduler.CurrentTick());
	m_unified_gpu_writes.Add(vaddr, size);
	m_unified_gpu_written.Add(vaddr, size);
}

void BufferCache::FinishPendingGpuWrites(uint64_t vaddr, uint64_t size) {
	if (HasPendingGpuWrites(vaddr, size)) {
		m_scheduler.Finish();
		m_pending_gpu_writes.Clear();
	}
}

bool BufferCache::HasPendingGpuWrites(uint64_t vaddr, uint64_t size) {
	if (m_pending_gpu_writes.Empty() || size == 0) {
		return false;
	}
	m_pending_gpu_writes.Expire(m_scheduler.GetMasterSemaphore().KnownGpuTick());
	return m_pending_gpu_writes.Intersects(vaddr, size);
}

// The recompiler reads a shader's tables of resources on the CPU when it records a draw or a dispatch. When a
// word it reads lies in a range the GPU may still write, the only safe value is the GPU's: submit everything and
// wait for it (Finish), which leaves the GPU dry while the thread records what follows. Games keep small counters
// there that a compute pass rewrites every frame to the same value (Astro Bot: two 32-byte counters, 40 waits an
// image, unchanged 5 196 times out of 5 218): once a word came out of eight waits unchanged, the read takes the
// value in memory without waiting, and the GPU's value is checked once its work is done. A difference revokes the
// word for good and says so on the console.
namespace {
constexpr uint32_t kSpeculationConfirmations = 8;
constexpr uint64_t kSpeculationRangeBytes    = 256; // a counter, not a table the GPU fills
} // namespace

void BufferCache::VerifySrtSpeculations() {
	const auto done = m_scheduler.GetMasterSemaphore().KnownGpuTick();
	for (auto& [address, speculation]: m_srt_speculation) {
		if (speculation.verify_tick == 0 || speculation.verify_tick > done) {
			continue;
		}
		speculation.verify_tick = 0;
		uint32_t landed = 0;
		if (!LibKernel::Memory::TryReadBacking(address, &landed, sizeof(landed)) ||
		    landed == speculation.used) {
			continue;
		}
		speculation.trusted = false;
		speculation.revoked = true;
		static uint32_t misses = 0;
		if (misses++ < 50) {
			std::printf("SRT speculation: the GPU wrote 0x%08x at 0x%016" PRIx64 " after 0x%08x was used; "
			            "this word waits again from now on\n",
			            landed, address, speculation.used);
		}
	}
}

bool BufferCache::FinishPendingGpuWritesUnder(std::span<const uint64_t> addresses) {
	if (m_pending_gpu_writes.Empty()) {
		return false;
	}
	m_pending_gpu_writes.Expire(m_scheduler.GetMasterSemaphore().KnownGpuTick());
	if (!m_srt_speculation.empty()) {
		VerifySrtSpeculations();
	}
	if (m_pending_gpu_writes.Empty()) {
		return false;
	}
	for (const auto address: addresses) {
		const auto* range = m_pending_gpu_writes.Find(address);
		if (range == nullptr) {
			continue;
		}
		auto& speculation = m_srt_speculation[address];
		if (speculation.trusted) {
			uint32_t used = 0;
			if (LibKernel::Memory::TryReadBacking(address, &used, sizeof(used))) {
				if (speculation.verify_tick == 0) {
					speculation.used = used;
				}
				speculation.verify_tick = std::max(speculation.verify_tick, range->tick);
				continue;
			}
		}
		uint32_t before = 0;
		LibKernel::Memory::TryReadBacking(address, &before, sizeof(before));
		const auto range_size = range->end - range->start;
		m_scheduler.Finish();
		m_pending_gpu_writes.Clear();
		if (!speculation.revoked && range_size <= kSpeculationRangeBytes) {
			uint32_t after = 0;
			LibKernel::Memory::TryReadBacking(address, &after, sizeof(after));
			if (after != before) {
				speculation.confirmations = 0;
			} else if (++speculation.confirmations >= kSpeculationConfirmations) {
				speculation.trusted = true;
			}
		}
		return true;
	}
	return false;
}

bool BufferCache::TakeUnifiedGpuWrites(uint64_t vaddr, uint64_t size) {
	if (!m_unified_gpu_writes.Intersects(vaddr, size)) {
		return false;
	}
	m_unified_gpu_writes.Subtract(vaddr, size);
	return true;
}

bool BufferCache::IsUnifiedGpuWritten(uint64_t vaddr, uint64_t size) const {
	return m_unified_gpu_written.Intersects(vaddr, size);
}

void BufferCache::ForgetUnifiedRange(uint64_t vaddr, uint64_t size) {
	if (!IsUnified() || size == 0) {
		return;
	}
	const auto begin = Common::AlignDown(vaddr, CACHING_PAGESIZE);
	const auto end   = Common::AlignUp(vaddr + size, CACHING_PAGESIZE);
	m_unified_gpu_writes.Subtract(begin, end - begin);
	m_unified_gpu_written.Subtract(begin, end - begin);
	auto       it    = m_unified_views.upper_bound(begin);
	if (it != m_unified_views.begin() && std::prev(it)->first + std::prev(it)->second.first > begin) {
		--it;
	}
	if (it == m_unified_views.end() || it->first >= end) {
		return;
	}
	BindUnifiedPieces(begin, end, {});
}

bool BufferCache::WriteGuestMemory(uint64_t vaddr, const void* data, uint64_t size) {
	if (!IsUnified() || ((vaddr | size) & 3u) != 0 || size == 0 || size > 65536) {
		return false;
	}
	const auto id = UnifiedWindow(vaddr, size);
	if (!id) {
		return false;
	}
	auto&      buffer = m_slot_buffers[id];
	const auto offset = buffer.Offset(vaddr);
	auto&      command = m_scheduler.Current();
	// A label is the console's end-of-pipe write: written once everything recorded before is done, without
	// stopping what follows. The buffer marker is that write; the low word first, so a reader waiting for a
	// growing value never sees it early. Only the CPU and WAIT_REG_MEM read labels: nothing to release.
	if (m_graphics.buffer_marker_enabled && (size == 4u || size == 8u)) {
		uint32_t words[2] {};
		std::memcpy(words, data, size);
		for (uint32_t i = 0; i < size / 4u; i++) {
			command.Handle().writeBufferMarker2AMD(vk::PipelineStageFlagBits2::eAllCommands, buffer.Handle(),
			                                       offset + i * 4u, words[i]);
		}
		uint64_t value = 0;
		std::memcpy(&value, data, size);
		m_pending_guest_writes[vaddr] = {size, m_scheduler.CurrentTick(), value};
		RecordUnifiedGpuWrite(vaddr, size);
		return true;
	}
	command.EndRendering();
	const auto native = command.Handle();
	// Everything recorded before lands first, as the GPU's own end-of-pipe write would.
	vk::MemoryBarrier before {};
	before.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 1, &before, 0, nullptr, 0,
	                       nullptr);
	native.updateBuffer(buffer.Handle(), offset, size, data);
	if (size <= sizeof(uint64_t)) {
		uint64_t value = 0;
		std::memcpy(&value, data, size);
		m_pending_guest_writes[vaddr] = {size, m_scheduler.CurrentTick(), value};
	}
	RecordUnifiedGpuWrite(vaddr, size);
	vk::MemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
	                      vk::AccessFlagBits::eHostRead;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands |
	                           vk::PipelineStageFlagBits::eHost,
	                       {}, 1, &after, 0, nullptr, 0, nullptr);
	return true;
}

std::pair<Buffer*, uint64_t> BufferCache::UnifiedGpuWriteTarget(uint64_t vaddr, uint64_t size) {
	if (!IsUnified()) {
		return {nullptr, 0};
	}
	const auto id = UnifiedWindow(vaddr, size);
	if (!id) {
		return {nullptr, 0};
	}
	auto& buffer = m_slot_buffers[id];
	ForgetColorMetadataChecks(vaddr, size);
	RecordUnifiedGpuWrite(vaddr, size);
	return {&buffer, buffer.Offset(vaddr)};
}

} // namespace Libs::Graphics
