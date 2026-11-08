set(KYTY_GIT_VERSION "unknown")
set(KYTY_GIT_HASH "unknown")
set(KYTY_GIT_REVISION "unknown")
if(GIT_EXECUTABLE)
	execute_process(
		COMMAND "${GIT_EXECUTABLE}" describe --tags --always --dirty
		WORKING_DIRECTORY "${GIT_WORKING_DIRECTORY}"
		OUTPUT_VARIABLE KYTY_GIT_VERSION
		OUTPUT_STRIP_TRAILING_WHITESPACE
		RESULT_VARIABLE GIT_RESULT
		ERROR_QUIET
	)
	if(NOT GIT_RESULT EQUAL 0)
		set(KYTY_GIT_VERSION "unknown")
	endif()

	execute_process(
		COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
		WORKING_DIRECTORY "${GIT_WORKING_DIRECTORY}"
		OUTPUT_VARIABLE KYTY_GIT_REVISION
		OUTPUT_STRIP_TRAILING_WHITESPACE
		RESULT_VARIABLE GIT_HASH_RESULT
		ERROR_QUIET
	)
	if(NOT GIT_HASH_RESULT EQUAL 0)
		set(KYTY_GIT_HASH "unknown")
		set(KYTY_GIT_REVISION "unknown")
	else()
		string(SUBSTRING "${KYTY_GIT_REVISION}" 0 7 KYTY_GIT_HASH)
		execute_process(
			COMMAND "${GIT_EXECUTABLE}" diff-index --quiet HEAD --
			WORKING_DIRECTORY "${GIT_WORKING_DIRECTORY}"
			RESULT_VARIABLE GIT_DIRTY_RESULT
			ERROR_QUIET
		)
		if(NOT GIT_DIRTY_RESULT EQUAL 0)
			string(APPEND KYTY_GIT_HASH "-dirty")
		endif()
	endif()
endif()
# GameFlex: what the shader store's records depend on - the recompiler, and the renderer that
# describes and creates pipelines (not its memory caches). A build that leaves these sources alone
# keeps the store, whatever else it changes.
set(KYTY_SHADER_STORE_STAMP "unknown")
if(GRAPHICS_SOURCE_DIR)
	file(GLOB_RECURSE store_sources LIST_DIRECTORIES false RELATIVE "${GRAPHICS_SOURCE_DIR}"
		"${GRAPHICS_SOURCE_DIR}/shader/*" "${GRAPHICS_SOURCE_DIR}/host_gpu/renderer/*")
	list(FILTER store_sources EXCLUDE REGEX "^host_gpu/renderer/cache/")
	list(SORT store_sources)
	set(store_digests "")
	foreach(source IN LISTS store_sources)
		file(SHA256 "${GRAPHICS_SOURCE_DIR}/${source}" digest)
		string(APPEND store_digests "${source} ${digest}\n")
	endforeach()
	if(store_sources)
		string(SHA256 KYTY_SHADER_STORE_STAMP "${store_digests}")
	endif()
endif()
# GameFlex: 0034 only translates what used to stop Kyty, and 0038 only moves when the store is read,
# so the records stored before them stay valid.
# Any other change to these sources gives another stamp, and a new store.
if(KYTY_SHADER_STORE_STAMP STREQUAL "58af55fba97c6709ae1c05a25943a503c1fe2bec899d2f11ba69c593a259f71c")
	set(KYTY_SHADER_STORE_STAMP "1a2216d3d966cf7b1b6e8ec85cd3a0a1a0f2115d9453cbf36d9a6c909aa85b7d")
endif()
configure_file("${INPUT_FILE}" "${OUTPUT_FILE}")
