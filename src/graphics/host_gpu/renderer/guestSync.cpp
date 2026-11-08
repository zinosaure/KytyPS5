#include "graphics/host_gpu/renderer/guestSync.h"

#include <algorithm>

namespace Libs::Graphics {

void GuestSync::Wait(vk::PipelineStageFlags2 stages, vk::AccessFlags2 access) {
	m_stages |= stages;
	m_access |= access;
	m_wait = true;
}

void GuestSync::Wrote(vk::PipelineStageFlags2 stages, vk::AccessFlags2 access) {
	m_stages |= stages;
	m_access |= access;
}

void GuestSync::WroteTransfer(vk::Buffer buffer, uint64_t offset, uint64_t size) {
	Wrote(vk::PipelineStageFlagBits2::eAllTransfer, vk::AccessFlagBits2::eTransferWrite);
	m_transfers.push_back({buffer, offset, size});
}

bool GuestSync::TransferOverlaps(vk::Buffer buffer, uint64_t offset, uint64_t size) const noexcept {
	return std::ranges::any_of(m_transfers, [&](const TransferRange& range) {
		return range.buffer == buffer && range.offset < offset + size && offset < range.offset + range.size;
	});
}

void GuestSync::Post(vk::CommandBuffer command) {
	if (static_cast<bool>(m_stages)) {
		vk::MemoryBarrier2 barrier {};
		barrier.srcStageMask  = m_stages;
		barrier.srcAccessMask = m_access;
		barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		// Everything a later command may read, and the writes it may make over the same memory.
		barrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite |
		                        vk::AccessFlagBits2::eUniformRead | vk::AccessFlagBits2::eIndirectCommandRead |
		                        vk::AccessFlagBits2::eVertexAttributeRead | vk::AccessFlagBits2::eIndexRead |
		                        vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite;
		if (m_access & (vk::AccessFlagBits2::eColorAttachmentWrite |
		                vk::AccessFlagBits2::eDepthStencilAttachmentWrite)) {
			barrier.dstAccessMask |= vk::AccessFlagBits2::eColorAttachmentRead |
			                         vk::AccessFlagBits2::eDepthStencilAttachmentRead;
		}
		vk::DependencyInfo dependency {};
		dependency.memoryBarrierCount = 1;
		dependency.pMemoryBarriers    = &barrier;
		command.pipelineBarrier2(dependency);
	}
	m_stages = {};
	m_access = {};
	m_wait   = false;
	m_transfers.clear();
}

void GuestSync::PostDue(vk::CommandBuffer command) {
	if (Due()) {
		Post(command);
	}
}

void GuestSync::BeforeTransfer(vk::CommandBuffer command, vk::Buffer buffer, uint64_t offset,
                               uint64_t size) {
	if (Due()) {
		Post(command);
		return;
	}
	if (!TransferOverlaps(buffer, offset, size)) {
		return;
	}
	vk::MemoryBarrier2 barrier {};
	barrier.srcStageMask  = vk::PipelineStageFlagBits2::eAllTransfer;
	barrier.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
	barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllTransfer;
	barrier.dstAccessMask = vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &barrier;
	command.pipelineBarrier2(dependency);
	m_transfers.clear();
}

} // namespace Libs::Graphics
