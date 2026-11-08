#version 450
#extension GL_ARB_shader_viewport_layer_array : require

// GameFlex: a quad over the whole target, one instance per slice of a volume. It stands in for the
// program a GPU without mesh shaders cannot run (PipelineCache::GetGraphicsPrograms): the pixel
// shader reads its position in the slice, and the slice.
layout(location = 0) out vec4 uv;

void main() {
	const vec2 corner = vec2(float(gl_VertexIndex & 1), float((gl_VertexIndex >> 1) & 1));
	gl_Position       = vec4(corner.x * 2.0 - 1.0, 1.0 - corner.y * 2.0, 0.0, 1.0);
	gl_Layer          = gl_InstanceIndex;
	uv                = vec4(corner, 0.0, 0.0);
}
