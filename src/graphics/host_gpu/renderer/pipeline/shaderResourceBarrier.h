#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SHADERRESOURCEBARRIER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SHADERRESOURCEBARRIER_H_

#include <atomic>
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

namespace Libs::Graphics {

struct ShaderStageRuntime;

vk::ShaderStageFlagBits NativeShaderStage(ShaderType stage);
vk::PipelineStageFlags  ShaderPipelineStages(vk::ShaderStageFlags stages);
vk::MemoryBarrier       MakeShaderWriteDependency();
vk::BufferMemoryBarrier MakeGdsDependency(vk::Buffer buffer);
bool HasShaderBufferWrites(const ShaderStageRuntime& runtime);
void ShaderAccessBarrier(vk::CommandBuffer vk_buffer, vk::PipelineStageFlags source_stages);
void ShaderWriteHazardBarrier(vk::CommandBuffer      vk_buffer,
                              vk::PipelineStageFlags destination_stages);
void ShaderWriteBarrier(vk::CommandBuffer vk_buffer, vk::PipelineStageFlags source_stages);

// GameFlex: grows at each synchronization the guest asks the GPU for (ACQUIRE_MEM, a partial flush, a cache
// writeback): GPU work on either side of it is ordered, work between two of them is not.
inline std::atomic<uint64_t> g_guest_sync_epoch {0};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SHADERRESOURCEBARRIER_H_
