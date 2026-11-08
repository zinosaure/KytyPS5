#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GUESTSYNC_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GUESTSYNC_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <vector>

namespace Libs::Graphics {

// The host GPU's synchronization follows the guest's.
//
// On the console, the draws, dispatches and DMA transfers of a queue overlap until the game asks the command
// processor to wait: ACQUIRE_MEM (idle and invalidate), a CS/VS/PS_PARTIAL_FLUSH (one stage done), a writeback of
// the color or depth caches, a DMA with CP_SYNC, a WAIT_REG_MEM on a label the GPU has yet to write. The renderer
// keeps one GuestSync for its command stream: the commands say what they wrote (Wrote), the command processor says
// when the game waits (Wait), and the next command posts one barrier for all of it (Post) — the stages the game
// named as the source, everything a later command may read or write as the destination. Nothing is ordered that
// the game did not order, as on the console; the end of a command buffer releases what the run still owes.
//
// Images keep their layout transitions, a host notion; the ordering of their writes comes from here as well.
class GuestSync {
public:
	GuestSync() = default;
	KYTY_CLASS_NO_COPY(GuestSync);

	// The game asked the GPU to finish `stages` (and to land the writes of `access`) before going on.
	void Wait(vk::PipelineStageFlags2 stages, vk::AccessFlags2 access = {});
	// A command wrote memory through `stages`: the next barrier releases it.
	void Wrote(vk::PipelineStageFlags2 stages, vk::AccessFlags2 access);
	// A transfer wrote `size` bytes of `buffer` at `offset`: the command after it releases it, and a later
	// transfer over the same bytes follows it, as the console's DMA engine runs its queue in order.
	void WroteTransfer(vk::Buffer buffer, uint64_t offset, uint64_t size);
	// Whether the game asked for a wait since the last barrier: one is due before the next command.
	[[nodiscard]] bool Due() const noexcept { return m_wait && static_cast<bool>(m_stages); }
	// Posts the barrier for everything pending, due or not, and forgets the run. Outside a render pass.
	void Post(vk::CommandBuffer command);
	// Posts the barrier due, if one is. Outside a render pass.
	void PostDue(vk::CommandBuffer command);
	// Before a transfer over `size` bytes of `buffer` at `offset`: the barrier for whatever the run wrote, or,
	// when only transfers wrote and one of them those bytes, a barrier between the two transfers.
	//
	// Transfers are the exception to following the game. Clair Obscur fills counters by DMA 6 500 times a
	// second and reads them from its shaders with no wait the command processor decodes, and the console gets
	// away with it; left unordered here, the GPU hung on a compute ring after two or three minutes, every
	// time, and never once with every transfer ordered with the commands around it. So a transfer waits for
	// the writes before it and the command after it waits for the transfer, until the packet the game relies
	// on is found.
	void BeforeTransfer(vk::CommandBuffer command, vk::Buffer buffer, uint64_t offset, uint64_t size);

	// Sync1 stage flags as sync2 flags: the bits coincide.
	[[nodiscard]] static vk::PipelineStageFlags2 Stages2(vk::PipelineStageFlags stages) noexcept {
		return vk::PipelineStageFlags2 {static_cast<uint64_t>(static_cast<uint32_t>(stages))};
	}

private:
	struct TransferRange {
		vk::Buffer buffer;
		uint64_t   offset;
		uint64_t   size;
	};

	[[nodiscard]] bool TransferOverlaps(vk::Buffer buffer, uint64_t offset, uint64_t size) const noexcept;

	vk::PipelineStageFlags2    m_stages {};
	vk::AccessFlags2           m_access {};
	bool                       m_wait = false; // the game asked to wait since the last barrier
	std::vector<TransferRange> m_transfers;    // the transfers of the run
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GUESTSYNC_H_
