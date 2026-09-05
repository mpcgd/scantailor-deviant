# Bundle the MuPDF shared library alongside the executables.
#
# Why: Homebrew's libmupdf.dylib reports compatibility version 0.0.0, so the
# dynamic loader will happily load a *new*, ABI-incompatible MuPDF into a
# previously built binary after `brew upgrade mupdf`. PDF loading then
# silently breaks until the project is rebuilt. Snapshotting the exact dylib
# that was linked against makes the binary immune to brew upgrades.
#
# Usage (run via `cmake -P` from a POST_BUILD step):
#   cmake -DSRC_LIB=<path to libmupdf.dylib> \
#         -DDEST_LIB=<where to place the bundled copy> \
#         -DBIN=<the executable just linked> \
#         -P bundle_mupdf.cmake
#
# The bundled copy is renamed to @rpath/libmupdf.dylib so that it resolves
# via the executables' existing @executable_path-based RPATH entries.

if(NOT EXISTS "${SRC_LIB}")
    message(FATAL_ERROR "bundle_mupdf: MuPDF library not found: ${SRC_LIB}")
endif()

set(BUNDLED_ID "@rpath/libmupdf.dylib")

# The install name (id) of the library the linker actually embedded into BIN.
execute_process(
    COMMAND /usr/bin/otool -D "${SRC_LIB}"
    OUTPUT_VARIABLE _otool_id_out
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _otool_id_result
)
if(NOT _otool_id_result EQUAL 0)
    message(FATAL_ERROR "bundle_mupdf: otool -D failed on ${SRC_LIB}")
endif()
string(REPLACE "\n" ";" _id_lines "${_otool_id_out}")
list(GET _id_lines 1 _src_id)

# 1. Snapshot the library (only re-copies when the source has changed).
get_filename_component(_dest_dir "${DEST_LIB}" DIRECTORY)
file(MAKE_DIRECTORY "${_dest_dir}")
execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different "${SRC_LIB}" "${DEST_LIB}")

# 2. Give the snapshot a stable, rpath-relative install name.
# Strip macOS xattrs first: files inside iCloud Drive folders carry
# com.apple.FinderInfo etc., which make codesign fail.
execute_process(COMMAND /usr/bin/xattr -cr "${DEST_LIB}" RESULT_VARIABLE _xattr_result)
execute_process(
    COMMAND /usr/bin/otool -D "${DEST_LIB}"
    OUTPUT_VARIABLE _dest_id_out
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REPLACE "\n" ";" _dest_id_lines "${_dest_id_out}")
list(GET _dest_id_lines 1 _dest_id)
if(NOT _dest_id STREQUAL BUNDLED_ID)
    execute_process(
        COMMAND /usr/bin/install_name_tool -id "${BUNDLED_ID}" "${DEST_LIB}"
        RESULT_VARIABLE _int_result
    )
    if(NOT _int_result EQUAL 0)
        message(FATAL_ERROR "bundle_mupdf: install_name_tool -id failed on ${DEST_LIB}")
    endif()
endif()
# Modifying the dylib invalidates its (ad-hoc) signature; macOS kills any
# process that maps an invalid page, so the copy must be re-signed. Always
# re-sign: copy_if_different may reuse a previous copy whose signature is
# already invalidated by a prior -id change.
execute_process(
    COMMAND /usr/bin/codesign --force --sign - "${DEST_LIB}"
    RESULT_VARIABLE _cs_lib_result
)
if(NOT _cs_lib_result EQUAL 0)
    message(WARNING "bundle_mupdf: ad-hoc re-sign of ${DEST_LIB} failed")
endif()

# 3. Point BIN at the snapshot instead of the Homebrew path.
execute_process(
    COMMAND /usr/bin/otool -L "${BIN}"
    OUTPUT_VARIABLE _otool_dep_out
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(_otool_dep_out MATCHES "${_src_id}")
    execute_process(
        COMMAND /usr/bin/install_name_tool -change "${_src_id}" "${BUNDLED_ID}" "${BIN}"
        RESULT_VARIABLE _change_result
    )
    if(NOT _change_result EQUAL 0)
        message(FATAL_ERROR "bundle_mupdf: install_name_tool -change failed on ${BIN}")
    endif()

    # 4. Modifying a Mach-O invalidates its (ad-hoc) signature; re-sign it.
    execute_process(
        COMMAND /usr/bin/codesign --force --sign - "${BIN}"
        RESULT_VARIABLE _cs_result
    )
    if(NOT _cs_result EQUAL 0)
        message(WARNING "bundle_mupdf: ad-hoc re-sign of ${BIN} failed")
    endif()

    # If BIN lives inside a .app bundle, the bundle seal must be refreshed too.
    string(FIND "${BIN}" ".app/Contents/MacOS/" _app_pos)
    if(NOT _app_pos EQUAL -1)
        string(SUBSTRING "${BIN}" 0 ${_app_pos} _bundle_dir)
        string(APPEND _bundle_dir ".app")
        # Detritus xattrs (e.g. from iCloud Drive) also break bundle signing.
        execute_process(COMMAND /usr/bin/xattr -cr "${_bundle_dir}")
        execute_process(
            COMMAND /usr/bin/codesign --force --sign - "${_bundle_dir}"
            RESULT_VARIABLE _cs_bundle_result
        )
        if(NOT _cs_bundle_result EQUAL 0)
            message(WARNING "bundle_mupdf: ad-hoc re-sign of ${_bundle_dir} failed")
        endif()
    endif()
endif()
