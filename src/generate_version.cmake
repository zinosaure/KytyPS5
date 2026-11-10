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
# GameFlex: what the shader store's records depend on - the recompiler, which builds the programs
# and their keys, and the code that describes and creates pipelines, with the render target headers
# it includes. A build that leaves these sources alone keeps the store, whatever else it changes:
# barriers, the scheduler, the draws, the caches of memory (they used to count, with all of
# host_gpu/renderer, and the Kyty revision on top: every build threw the accounts' stores away).
set(KYTY_SHADER_STORE_STAMP "unknown")
if(GRAPHICS_SOURCE_DIR)
	file(GLOB_RECURSE store_sources LIST_DIRECTORIES false RELATIVE "${GRAPHICS_SOURCE_DIR}"
		"${GRAPHICS_SOURCE_DIR}/shader/*" "${GRAPHICS_SOURCE_DIR}/host_gpu/renderer/pipeline/*")
	list(APPEND store_sources "host_gpu/renderer/colorRenderTarget.h"
		"host_gpu/renderer/depthRenderTarget.h" "host_gpu/renderer/renderTarget.h")
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
# Any change to these sources gives another stamp, and a new store.
configure_file("${INPUT_FILE}" "${OUTPUT_FILE}")
