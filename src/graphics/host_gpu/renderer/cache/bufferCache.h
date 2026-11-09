#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/faultManager.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/cache/pendingGpuWrites.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <array>
#include <map>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <mutex>
#include <span>
#include <utility>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;
class TextureCache;

using BufferId = Common::SlotId;
inline constexpr BufferId NULL_BUFFER_ID {0};

class BufferCache {
public:
	static constexpr uint32_t CACHING_PAGEBITS  = 14;
	static constexpr uint64_t CACHING_PAGESIZE  = uint64_t {1} << CACHING_PAGEBITS;
	static constexpr uint64_t CACHING_NUMPAGES  = (LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemorySize) >> CACHING_PAGEBITS;
	static constexpr uint64_t BDA_PAGETABLE_SIZE =
	    CACHING_NUMPAGES * sizeof(vk::DeviceAddress);

	static constexpr uint64_t PageIndex(uint64_t address) {
		return (address < LOWER_ADDRESS_SIZE
		            ? address
		            : address - LibKernel::Memory::kExtendedMemoryBase + LOWER_ADDRESS_SIZE) >>
		       CACHING_PAGEBITS;
	}
	static constexpr uint64_t GuestAddress(uint64_t offset) {
		return offset < LOWER_ADDRESS_SIZE
		           ? offset
		           : offset - LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemoryBase;
	}

	BufferCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	            TextureCache& texture_cache);
	~BufferCache();
	KYTY_CLASS_NO_COPY(BufferCache);

	void                   InvalidateMemory(uint64_t vaddr, uint64_t size);
	void                   ReadMemory(uint64_t vaddr, uint64_t size, bool is_write = false);
	[[nodiscard]] Buffer&  GetBuffer(BufferId id) { return m_slot_buffers[id]; }
	[[nodiscard]] BufferId FindBuffer(uint64_t vaddr, uint64_t size);
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBuffer(uint64_t vaddr, uint64_t size,
	                                                        bool     is_written,
	                                                        bool     is_texel_buffer = false,
	                                                        BufferId id              = {});
	[[nodiscard]] StreamBuffer&                GetUtilityBuffer(MemoryUsage usage) noexcept {
		switch (usage) {
			case MemoryUsage::Upload: return m_staging_buffer;
			case MemoryUsage::Stream: return m_stream_buffer;
			case MemoryUsage::Download: return m_download_buffer;
			case MemoryUsage::DeviceLocal: return m_device_buffer;
		}
		EXIT("BufferCache: invalid utility-buffer usage\n");
	}
	[[nodiscard]] const Buffer* GetGdsBuffer() const noexcept { return &m_gds_buffer; }
	[[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept { return &m_bda_pagetable_buffer; }
	[[nodiscard]] Buffer* GetFaultBuffer() noexcept { return m_fault_manager.GetFaultBuffer(); }
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferForImage(uint64_t vaddr, uint64_t size);
	void FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds);
	// Unified memory: the guest's memory is imported, and buffers over it are views of it,
	// with nothing to upload, protect or download.
	[[nodiscard]] bool IsUnified() const noexcept { return m_unified_memory != nullptr; }
	// Writes guest memory in command order (a label, WRITE_DATA): false outside unified memory,
	// where the caller writes it itself.
	bool WriteGuestMemory(uint64_t vaddr, const void* data, uint64_t size);
	// Unified memory: the buffer and offset through which GPU work writes [vaddr, vaddr + size) of
	// the guest's memory, recorded as GPU-written; nullptr outside unified memory.
	[[nodiscard]] std::pair<Buffer*, uint64_t> UnifiedGpuWriteTarget(uint64_t vaddr, uint64_t size);
	// The value of a label WriteGuestMemory recorded at exactly [vaddr, vaddr + size), that the
	// GPU has not written yet: work recorded after it sees it, in command order.
	[[nodiscard]] bool PendingGuestWrite(uint64_t vaddr, uint64_t size, uint64_t* value);
	// Unified memory: Kyty's own reads at record time (indirect arguments, shader constants)
	// wait for GPU work recorded before them that may write [vaddr, vaddr + size). The guest's
	// threads never do: they wait on the GPU's labels, as on the console.
	void FinishPendingGpuWrites(uint64_t vaddr, uint64_t size);
	// Unified memory: whether GPU work recorded and not done yet may write [vaddr, vaddr + size).
	[[nodiscard]] bool HasPendingGpuWrites(uint64_t vaddr, uint64_t size);
	// As FinishPendingGpuWrites, for the 4-byte reads of the SRT walker; true when it waited.
	bool FinishPendingGpuWritesUnder(std::span<const uint64_t> addresses);
	// A word of a table of resources inside a range the GPU may still write, read without waiting once its value
	// came out of eight waits unchanged; the GPU's value is compared afterwards. A difference withdraws the trust,
	// which comes back after twice as many unchanged waits each time; the fifth revokes the word for good.
	struct SrtSpeculation {
		uint32_t confirmations = 0; // consecutive waits that changed nothing
		bool     trusted       = false;
		bool     revoked       = false; // wrong five times: never trusted again
		uint32_t misses        = 0;     // differences seen so far
		uint32_t used          = 0;     // the value read without waiting
		uint64_t verify_tick   = 0;     // the GPU work it skipped waiting for; 0 when nothing is to verify
	};
	void VerifySrtSpeculations();
	// GameFlex: a guest wait (WAIT_REG_MEM) for a value a shader recorded before will write. The GPU runs the
	// stream in order: behind a barrier, everything recorded after the wait runs after that shader, which is all
	// the wait asks, as long as the shader writes the value. An address is trusted once four waits there ended
	// with the value. At each trusted wait the GPU copies the value at that place in the stream, and the copy
	// is checked once its work is done: a miss revokes the address for good. True: the wait may go on.
	[[nodiscard]] bool TrustPendingGpuWrite(uint64_t vaddr, uint64_t size, uint32_t func, uint64_t ref, uint64_t mask);
	// The guest wait at vaddr found its value in memory.
	void GuestWaitPassed(uint64_t vaddr);
	// Unified memory: whether the GPU wrote [vaddr, vaddr + size) since the last call for it. GPU
	// writes leave no fault: a reader that keeps its own copy (the image heap) asks.
	[[nodiscard]] bool TakeUnifiedGpuWrites(uint64_t vaddr, uint64_t size);
	// Unified memory: whether the GPU ever wrote [vaddr, vaddr + size), as the old path's
	// GPU-modified state says until a download.
	[[nodiscard]] bool IsUnifiedGpuWritten(uint64_t vaddr, uint64_t size) const;
	// The guest unmapped or remapped [vaddr, vaddr + size): views of it must go.
	void ForgetUnifiedRange(uint64_t vaddr, uint64_t size);
	void CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
	                bool src_gds);
	// Cache-index and exact dirty-range queries require GPU-thread serialization.
	[[nodiscard]] bool IsRegionRegistered(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool HasGpuDirtyBytes(uint64_t vaddr, uint64_t size);
	// How many of these bytes the GPU wrote and the guest memory does not hold yet, at most size.
	[[nodiscard]] uint64_t GpuDirtyBytes(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	// Color metadata the GPU found free of the clear keys in `keys` (bits of code >> 5) stays so
	// until a GPU write or download.
	void               MarkColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys);
	[[nodiscard]] bool IsColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys) const;
	void               ProcessFaultBuffer();
	void               SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size);
	// Queue after the range's state change becomes visible, or a drain could consume it first.
	void                   QueueBdaSync(uint64_t vaddr, uint64_t size);
	[[nodiscard]] RangeSet TakeBdaSync();
	void               RunGarbageCollector();

private:
	friend struct BufferCacheTestAccess;

	bool IsBufferInvalid(BufferId id) const { return !id || m_slot_buffers[id].is_deleted; }

	using BufferMap = std::map<uint64_t, BufferId>;
	struct OverlapResult {
		BufferMap::iterator first;
		BufferMap::iterator last;
		uint64_t            begin;
		uint64_t            end;
		bool                has_stream_leap;
	};

	using PageTable = MultiLevelPageTable<BufferId, CACHING_PAGEBITS, 44, 20>;
	static_assert(CACHING_PAGESIZE == (uint64_t {1} << PageTable::kPageBits));
	void WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source, uint64_t size);
	void TouchBuffer(const Buffer& buffer);
	[[nodiscard]] OverlapResult ResolveOverlaps(uint64_t vaddr, uint64_t size);
	void JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score);
	[[nodiscard]] BufferId CreateBuffer(uint64_t vaddr, uint64_t size);
	void ImportUnifiedMemory();
	// The window over [vaddr, vaddr + size) with the guest's views bound in it, or nothing when
	// the range is not wholly guest memory (modules) or spans more than a window.
	[[nodiscard]] BufferId UnifiedWindow(uint64_t vaddr, uint64_t size);
	// Binds the guest's current views over [vaddr, vaddr + size) in the windows that cover it,
	// and points the BDA table at them. True when every page of the range is guest memory.
	bool BindUnified(uint64_t vaddr, uint64_t size);
	// GPU work recorded now writes [vaddr, vaddr + size) of the guest memory.
	void RecordUnifiedGpuWrite(uint64_t vaddr, uint64_t size);
	void BindUnifiedPieces(uint64_t begin, uint64_t end,
	                       const std::vector<std::tuple<uint64_t, uint64_t, uint64_t>>& pieces);
	void                   Register(BufferId id);
	void Unregister(BufferId id);
	template <bool insert>
	void ChangeRegister(BufferId id);
	void DeleteBuffer(BufferId id);
	[[nodiscard]] bool SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                     bool is_written, bool is_texel_buffer);
	[[nodiscard]] vk::Buffer UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
	                                      uint64_t total_size);
	[[nodiscard]] bool SynchronizeBufferFromImage(Buffer& buffer, uint64_t vaddr, uint64_t size);
	// Synchronous downloads publish before returning; asynchronous callers wait before reuse.
	template <bool async>
	[[nodiscard]] bool DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size);
	void               ForgetColorMetadataChecks(uint64_t vaddr, uint64_t size);

	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	FaultManager                                      m_fault_manager;
	Buffer                                            m_gds_buffer;
	Buffer                                            m_bda_pagetable_buffer;
	Common::SlotVector<Buffer>                        m_slot_buffers;
	Common::LeastRecentlyUsedCache<BufferId, uint64_t> m_lru_cache;
	BufferMap                                         m_buffers;
	PageTable                                         m_page_table;
	RangeSet                                          m_gpu_modified_ranges;
	// One per TileManager::ColorClearSlots key slot.
	std::array<RangeSet, 8>                           m_checked_color_metadata;
	MemoryTracker                                     m_memory_tracker;
	std::mutex                                        m_bda_sync_mutex;
	RangeSet                                          m_bda_sync;
	StreamBuffer                                      m_staging_buffer;
	StreamBuffer                                      m_stream_buffer;
	StreamBuffer                                      m_download_buffer;
	StreamBuffer                                      m_device_buffer;
	TextureCache&                                     m_texture_cache;
	uint64_t                                          m_total_used_memory  = 0;
	uint64_t m_trigger_gc_memory  = 1ull * 1024 * 1024 * 1024;
	uint64_t m_critical_gc_memory = 2ull * 1024 * 1024 * 1024;
	uint64_t m_gc_tick            = 0;
	vk::DeviceMemory                                  m_unified_memory {};
	uint8_t*                                          m_unified_base = nullptr;
	// Labels the GPU writes, by address.
	struct PendingGuestWriteEntry {
		uint64_t size  = 0;
		uint64_t tick  = 0;
		uint64_t value = 0;
	};
	std::map<uint64_t, PendingGuestWriteEntry> m_pending_guest_writes;
	uint64_t                                   m_pending_guest_writes_expired = 0; // the GPU tick of the last expiry
	// A window of guest addresses starts at every GiB and covers two: any range up to 1 GiB fits
	// one (the device's buffers stay under 4 GiB).
	static constexpr uint32_t UnifiedWindowBits = 30;
	static constexpr uint64_t UnifiedWindowSize = uint64_t {2} << UnifiedWindowBits;
	std::unordered_map<uint64_t, BufferId>            m_unified_windows;
	// The guest views bound in the windows: address -> (size, offset in the dma-buf).
	std::map<uint64_t, std::pair<uint64_t, uint64_t>> m_unified_views;
	vk::Fence                                         m_unified_bind_fence {};
	// The ranges bound for writing in command buffers the GPU has not finished.
	PendingGpuWrites                                  m_pending_gpu_writes;
	std::unordered_map<uint64_t, SrtSpeculation>      m_srt_speculation;
	// The words and addresses whose prediction waits to be checked, with the GPU tick in their entry: the checks
	// walk these, not every address ever seen (a ring of flags grows the maps for good).
	std::vector<uint64_t>                             m_srt_verifications;
	struct FlagSpeculation {
		uint32_t confirmations = 0;
		bool     waiting       = false; // suspended on a pending shader write, value not found yet
		bool     trusted       = false;
		bool     revoked       = false; // once wrong, never trusted again
	};
	// A trusted wait: the condition, and the slot of m_flag_captures the GPU copies the value to in the command
	// buffer of `tick`, at the wait's place in the stream.
	struct FlagCheck {
		uint64_t address = 0;
		uint64_t tick    = 0;
		uint64_t ref     = 0;
		uint64_t mask    = 0;
		uint32_t func    = 0;
		uint32_t size    = 0;
		uint32_t slot    = 0;
	};
	std::vector<FlagCheck>                            m_flag_verifications;
	std::unique_ptr<Buffer>                           m_flag_captures;
	uint32_t                                          m_flag_capture_next = 0;
	void VerifyFlagSpeculations();
	std::unordered_map<uint64_t, FlagSpeculation>     m_flag_speculation;
	// Ranges bound for writing that no TakeUnifiedGpuWrites has taken yet, and all of them.
	RangeSet                                          m_unified_gpu_writes;
	RangeSet                                          m_unified_gpu_written;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
