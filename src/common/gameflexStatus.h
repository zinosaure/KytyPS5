#ifndef KYTY_COMMON_GAMEFLEX_STATUS_H_
#define KYTY_COMMON_GAMEFLEX_STATUS_H_

#include <atomic>
#include <cstdint>
#include <cstdlib>

namespace Common::GameFlex {

// GameFlex: the scene size Kyty asks an Unreal game for (r.ScreenPercentage from KYTY_UE_RENDER_HEIGHT),
// width << 32 | height, 0 when it asks for none. The window title carries it for GameFlex's
// performance panel, next to the image the game outputs.
inline std::atomic<uint64_t> g_unreal_scene_size {0};

// GameFlex: the resolution of the console's video output, a setting as on a PS5 (KYTY_OUTPUT_RESOLUTION): 1080, the
// default and the smallest a PS5 offers, or 2160. The console announces it to every game, and an Unreal game gets it
// as the size of the images it hands to the console (PS5.DefaultBackBufferWidth/Height), since that engine never asks.
inline uint32_t ConsoleOutputHeight() {
	static const uint32_t height = [] {
		const char* value = std::getenv("KYTY_OUTPUT_RESOLUTION");
		return value != nullptr && std::strtoul(value, nullptr, 10) == 2160 ? 2160u : 1080u;
	}();
	return height;
}

} // namespace Common::GameFlex

#endif /* KYTY_COMMON_GAMEFLEX_STATUS_H_ */
