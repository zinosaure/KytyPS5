#include "graphics/host_gpu/renderer/guestSync.h"
#include "common/commandThreadProfile.h"

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
	m_wait = true; // the command after a transfer waits for it (see BeforeTransfer)
}

bool GuestSync::TransferOverlaps(vk::Buffer buffer, uint64_t offset, uint64_t size) const noexcept {
	return std::ranges::any_of(m_transfers, [&](const TransferRange& range) {
		return range.buffer == buffer && range.offset < offset + size && offset < range.offset + range.size;
	});
}

void GuestSync::Post(vk::CommandBuffer command) {
	if (static_cast<bool>(m_stages)) {
		Common::CommandThreadProfile::Scope ctp_barrier(Common::CommandThreadProfile::Cause::Barriers);
		Common::CommandThreadProfile::Count(Common::CommandThreadProfile::Event::Barriers);
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
	// Shaders wrote since the last barrier: a transfer waits for them. Transfers alone: ordered below, only
	// over the same bytes.
	const bool transfers_only =
	    (m_stages & ~vk::PipelineStageFlags2 {vk::PipelineStageFlagBits2::eAllTransfer}) == vk::PipelineStageFlags2 {};
	if (Due() || (static_cast<bool>(m_stages) && !transfers_only)) {
		Post(command);
		return;
	}
	if (!TransferOverlaps(buffer, offset, size)) {
		return;
	}
	Common::CommandThreadProfile::Scope ctp_profile(Common::CommandThreadProfile::Cause::Barriers);
	Common::CommandThreadProfile::Count(Common::CommandThreadProfile::Event::Barriers);
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
