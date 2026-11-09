#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <span>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

namespace HW {
class Context;
class UserConfig;
class Shader;
} // namespace HW

struct GraphicContext;
struct DrawCallInfo;
struct DrawEmitInfo;
struct DrawIndexBufferSource;
class RenderContext;
class CommandScheduler;
struct RenderExecutorTestAccess;

enum class DrawOffsetSource : uint8_t {
	DrawState,
	IndirectArgs,
};

// Nonzero address: the GPU reads count Vulkan indirect draw records, stride bytes apart.
struct DrawIndirectArgs {
	uint64_t              address = 0;
	uint32_t              count   = 0;
	uint32_t              stride  = 0;
	ShaderIndirectOffsets offsets;
	// The guest's count, read by the GPU, of at most `count` records; 0 when `count` is it.
	uint64_t              count_address = 0;
};

struct DrawIndexArgs {
	uint32_t         index_count         = 0;
	const void*      index_addr          = nullptr;
	uint32_t         instance_count      = 0;
	uint32_t         index_type_and_size = 0;
	int32_t          base_vertex         = 0;
	uint32_t         first_instance      = 0;
	DrawOffsetSource offset_source       = DrawOffsetSource::DrawState;
	// With indirect records, index_count is INDEX_BUFFER_SIZE.
	DrawIndirectArgs indirect;
};

struct DrawAutoArgs {
	uint32_t         vertex_count   = 0;
	uint32_t         instance_count = 0;
	uint32_t         first_vertex   = 0;
	uint32_t         first_instance = 0;
	DrawOffsetSource offset_source  = DrawOffsetSource::DrawState;
	DrawIndirectArgs indirect;
};

struct SubmitInfo {
	static constexpr uint32_t MaxSemaphores = 3;

	std::array<vk::Semaphore, MaxSemaphores>          wait_semaphores {};
	std::array<uint64_t, MaxSemaphores>               wait_ticks {};
	std::array<vk::PipelineStageFlags, MaxSemaphores> wait_stages {};
	std::array<vk::Semaphore, MaxSemaphores>          signal_semaphores {};
	std::array<uint64_t, MaxSemaphores>               signal_ticks {};
	uint32_t                                          num_wait_semaphores   = 0;
	uint32_t                                          num_signal_semaphores = 0;

	void AddWait(vk::Semaphore semaphore, uint64_t tick = 1,
	             vk::PipelineStageFlags stage = vk::PipelineStageFlagBits::eAllCommands) {
		EXIT_IF(semaphore == nullptr || num_wait_semaphores >= MaxSemaphores);
		wait_semaphores[num_wait_semaphores] = semaphore;
		wait_ticks[num_wait_semaphores]      = tick;
		wait_stages[num_wait_semaphores++]   = stage;
	}

	void AddSignal(vk::Semaphore semaphore, uint64_t tick = 1) {
		EXIT_IF(semaphore == nullptr || num_signal_semaphores >= MaxSemaphores);
		signal_semaphores[num_signal_semaphores] = semaphore;
		signal_ticks[num_signal_semaphores++]    = tick;
	}
};

class CommandBuffer {
public:
	~CommandBuffer() = default;

	KYTY_CLASS_NO_COPY(CommandBuffer);

	[[nodiscard]] bool IsInvalid() const noexcept { return m_buffer == nullptr; }

	void BeginRendering(const RenderState& state) const;
	void EndRendering() const;
	// Ends the render pass and posts the barrier the game asked for, when one is due (GuestSync).
	void PostGuestBarrier() const;

	[[nodiscard]] vk::CommandBuffer Handle() const noexcept { return m_buffer; }
	[[nodiscard]] GraphicContext&   GetGraphics() const noexcept { return m_graphics; }
	[[nodiscard]] RenderContext&    GetContext() const noexcept { return m_context; }
	[[nodiscard]] HW::Context&      GetRegisters() const noexcept { return *m_registers; }
	[[nodiscard]] HW::UserConfig&   GetUserConfig() const noexcept { return *m_user_config; }
	[[nodiscard]] HW::Shader&       GetShaders() const noexcept { return *m_shaders; }

private:
	explicit CommandBuffer(CommandScheduler& scheduler);
	void Bind(HW::Context& registers, HW::UserConfig& user_config, HW::Shader& shaders) noexcept {
		m_registers   = &registers;
		m_user_config = &user_config;
		m_shaders     = &shaders;
	}

	void Begin();
	void End() const;

	RenderContext&      m_context;
	GraphicContext&     m_graphics;
	vk::CommandBuffer   m_buffer = nullptr;
	mutable RenderState m_render_state;
	mutable bool        m_rendering   = false;
	HW::Context*        m_registers   = nullptr;
	HW::UserConfig*     m_user_config = nullptr;
	HW::Shader*         m_shaders     = nullptr;

	friend class CommandScheduler;
};

namespace Pssr {
struct Request;
}

class RenderExecutor {
public:
	explicit RenderExecutor(RenderContext& context): m_context(context) {}
	~RenderExecutor();
	KYTY_CLASS_NO_COPY(RenderExecutor);

	void DispatchDirect(CommandBuffer& buffer, uint32_t thread_group_x, uint32_t thread_group_y,
	                    uint32_t thread_group_z, uint32_t mode, bool async_compute);
	void DispatchIndirect(CommandBuffer& buffer, uint64_t args_addr, uint32_t mode,
	                      bool async_compute);
	void ProcessPssr(CommandBuffer& buffer, const Pssr::Request& request);

	void PrepareBindings(const ShaderStageRuntime& runtime, PreparedBindings& prepared);
	void                           FindBuffers(std::span<PreparedBindings* const> stages);
	void                           RebindBuffers(PreparedBindings& bindings);
	void                           RebindImages(PreparedBindings& bindings);
	void CommitBindings(CommandBuffer& buffer, vk::PipelineBindPoint pipeline_bind_point,
	                    const PipelineCache::Pipeline&     pipeline,
	                    std::span<PreparedBindings* const> bindings);
	[[nodiscard]] TextureBinding ResolveTexture(const ShaderRecompiler::IR::ImageResource&   resource,
	                                            const ShaderRecompiler::IR::DescriptorValue& value);

private:
	// False, with nothing drawn, when the draw's indirect records need CPU counts.
	bool DrawIndex(CommandBuffer& buffer, const DrawIndexArgs& args);
	bool DrawAuto(CommandBuffer& buffer, const DrawAutoArgs& args);

	struct GraphicsBindings {
		std::array<PreparedBindings, 3> vertex;
		PreparedBindings                pixel;
	};

	void PrepareGraphicsBindings(std::span<PreparedBindings* const> stages,
	                             std::span<RenderColorInfo> colors);
	void ResolveRenderColorTarget(CommandBuffer& buffer, RenderColorInfo& target,
	                              uint32_t render_target_slot, bool ignore_target_mask = false,
	                              bool exact_format = false);
	void ResolveRenderDepthTarget(CommandBuffer& buffer, RenderDepthInfo& target);
	[[nodiscard]] bool DepthStencilCopy(CommandBuffer& buffer);
	[[nodiscard]] bool CopyDepthToColor(CommandBuffer& buffer, const DrawCallInfo& draw,
	                                   const DrawEmitInfo& emit, DrawRenderState& state);
	[[nodiscard]] bool PrepareDrawRenderState(CommandBuffer& buffer, const DrawCallInfo& draw,
	                                          DrawRenderState& state);
	void                      ExecutePreparedDraw(CommandBuffer& buffer, const DrawCallInfo& draw,
	                                              DrawRenderState& state, const DrawEmitInfo& emit,
	                                              const DrawIndexBufferSource& index_source);
	[[nodiscard]] RenderState AcquireRenderTargets(CommandBuffer& buffer, RenderColorInfo* colors,
	                                               uint32_t color_count, RenderDepthInfo& depth,
	                                               vk::ImageAspectFlags& feedback_aspects,
	                                               std::span<PreparedBindings* const> stages = {});
	[[nodiscard]] bool        ResolveColorTargets(CommandBuffer& buffer);
	[[nodiscard]] bool        ConsumeMetadataColorOperation(CommandBuffer& buffer);
	void                      BindImage(ImageId id, bool storage);
	void                      BindRenderTarget(ImageId id);
	void                      ResetBindings();
	[[nodiscard]] bool        TryConsumeComputeMetaClear(const ShaderComputeInputInfo& input,
	                                                     const CommandBuffer&          buffer);
	[[nodiscard]] bool TryConsumeComputeFill(const ShaderComputeInputInfo& input,
	                                        CommandBuffer& command, uint32_t group_x,
	                                        uint32_t group_y, uint32_t group_z, uint32_t mode);

	// The texture bound for a descriptor, found once. ResolveTexture decodes the descriptor, computes the
	// image's layout and searches the cache by pages at every binding of every draw (12 % of the command
	// thread in Astro Bot), for bindings that repeat image after image: the same descriptor, for the same
	// shader resource, binds the same image as long as the texture cache registered or freed nothing over
	// the memory it describes since (the region's generation), when that image was alone over its memory.
	struct TextureBindingKey {
		std::array<uint32_t, 8>                     dwords;
		const ShaderRecompiler::IR::ImageResource* resource;
		bool operator==(const TextureBindingKey&) const = default;
	};
	struct TextureBindingKeyHash {
		size_t operator()(const TextureBindingKey& key) const noexcept;
	};
	struct CachedTextureBinding {
		ImageId                 found_id; // what FindImage returned
		ImageId                 image_id; // what is bound (the depth owner of a stencil plane)
		TextureCache::ImageDesc desc;
		uint32_t                metadata_base_layer;
		uint64_t                generation;
	};

	// The color target the draw before found in a slot. Astro Bot's draws repeat their targets 98.7 % of the time,
	// and finding a target, touching it and checking its metadata clears cost 6.5 % of the command thread at every
	// draw. A draw takes it again as it is when nothing but register writes came between the two draws: the same
	// command buffer, the same registers, no image registered or freed, no transfer, no dispatch or image operation
	// (m_chain_breaks). The draw before touched it in this command buffer and checked its metadata.
	struct ContinuedTarget {
		HW::RenderTarget registers;
		uint32_t         mask         = 0;
		bool             exact_format = false;
		bool             valid        = false;
		uint64_t         tick         = 0;
		uint64_t         textures     = 0; // TextureCache::Epoch
		uint64_t         transfers    = 0; // BufferCache::Transfers
		uint64_t         breaks       = 0; // m_chain_breaks
		RenderColorInfo  info;
	};

	RenderContext&                        m_context;
	std::unordered_map<TextureBindingKey, CachedTextureBinding, TextureBindingKeyHash> m_texture_bindings;
	std::array<ContinuedTarget, RENDER_COLOR_ATTACHMENTS_MAX> m_continued_targets;
	// Advances at every dispatch and at every draw spent on an image operation (metadata, copy, resolve).
	uint64_t                                                  m_chain_breaks = 0;
	GraphicsBindings                     m_graphics_bindings;
	PreparedBindings                     m_compute_bindings;
	std::vector<ImageId>                  m_bound_images;
	std::vector<vk::DescriptorImageInfo>  m_descriptor_images;
	std::vector<vk::WriteDescriptorSet>   m_descriptor_writes;
	vk::DescriptorSetLayout               m_pssr_descriptor_layout = nullptr;
	vk::PipelineLayout                    m_pssr_pipeline_layout   = nullptr;
	vk::Pipeline                          m_pssr_pipeline          = nullptr;

	friend class CommandProcessor;
	friend struct RenderExecutorTestAccess;
};

[[nodiscard]] bool ResolveComputeBufferFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                                            uint32_t group_y, uint32_t group_z, uint32_t mode,
                                            ShaderBufferResource& descriptor,
                                            uint32_t& packed_clear, uint64_t& size);

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_ */
