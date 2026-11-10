#pragma once

#include "graphics/shader/recompiler/ir/Block.h"
#include "graphics/shader/shader.h"

namespace Libs::Graphics::ShaderRecompiler::IR {

// The stage lets a vertex shader's entry lane count ground lane-mask tests.
void ConstantPropagationPass(const BlockList& blocks, uint32_t wave_size = 64,
                             ShaderType stage = ShaderType::Unknown);

} // namespace Libs::Graphics::ShaderRecompiler::IR
