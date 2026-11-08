#ifndef KYTY_COMMON_GAMEFLEX_STATUS_H_
#define KYTY_COMMON_GAMEFLEX_STATUS_H_

#include <atomic>
#include <cstdint>

namespace Common::GameFlex {

// GameFlex: the scene size Kyty asks an Unreal game for (r.ScreenPercentage from KYTY_UE_RENDER_HEIGHT),
// width << 32 | height, 0 when it asks for none. The window title carries it for GameFlex's
// performance panel, next to the image the game outputs.
inline std::atomic<uint64_t> g_unreal_scene_size {0};

} // namespace Common::GameFlex

#endif /* KYTY_COMMON_GAMEFLEX_STATUS_H_ */
