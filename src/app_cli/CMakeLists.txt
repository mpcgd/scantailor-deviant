
INCLUDE_DIRECTORIES("${CMAKE_SOURCE_DIR}/src/core")

SET(
        cli_only_sources
        ConsoleBatch.cpp ConsoleBatch.h
        main-cli.cpp
)

ADD_EXECUTABLE(scantailor-deviant-cli ${cli_only_sources} )

IF(APPLE)
        SET_TARGET_PROPERTIES(scantailor-deviant-cli PROPERTIES
                INSTALL_RPATH "@executable_path/../Frameworks"
                BUILD_RPATH   "@executable_path/../Frameworks;@executable_path/../../Frameworks"
        )
        IF(MuPDF_LIBRARY MATCHES "\\.dylib$")
                # Snapshot the exact MuPDF dylib we linked against so that
                # `brew upgrade mupdf` cannot break this binary.
                ADD_CUSTOM_COMMAND(TARGET scantailor-deviant-cli POST_BUILD
                        COMMAND ${CMAKE_COMMAND}
                                -DSRC_LIB=${MuPDF_LIBRARY}
                                -DDEST_LIB=${CMAKE_BINARY_DIR}/Frameworks/libmupdf.dylib
                                -DBIN=$<TARGET_FILE:scantailor-deviant-cli>
                                -P ${PROJECT_SOURCE_DIR}/cmake/bundle_mupdf.cmake
                )
        ENDIF()
ENDIF()

TARGET_LINK_LIBRARIES(
        scantailor-deviant-cli
        fix_orientation page_split deskew select_content page_layout output stcore
	dewarping zones interaction imageproc math foundation exporting
)

IF(OPENMP_FOUND)
        TARGET_LINK_LIBRARIES(scantailor-deviant-cli OpenMP::OpenMP_CXX)
ENDIF()

# Widgets module is used statically but not at runtime.
TARGET_LINK_LIBRARIES(scantailor-deviant-cli Qt5::Widgets Qt5::Xml)

IF(EXTRA_LIBS)
        TARGET_LINK_LIBRARIES(scantailor-deviant-cli ${EXTRA_LIBS})
ENDIF()

IF(APPLE)
        INSTALL(TARGETS scantailor-deviant-cli RUNTIME DESTINATION scantailor-deviant.app/Contents/Library)
ELSEIF(WIN32)
        INSTALL(TARGETS scantailor-deviant-cli RUNTIME DESTINATION .)
ELSE()
        INSTALL(TARGETS scantailor-deviant-cli RUNTIME DESTINATION bin)
ENDIF()

TRANSLATION_SOURCES(
        scantailor-deviant
        ${cli_only_sources}
)
