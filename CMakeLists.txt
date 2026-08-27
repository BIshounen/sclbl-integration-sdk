cmake_minimum_required(VERSION 3.10.2)

project(nxai-integration-examples)

# Add Scailable C Utilities for all subprojects. Linux/WSL only: this vendored
# copy is POSIX-only (sysv_ipc-based C sources) and does not build on Windows;
# postprocessor-python-clip-situation below doesn't need it (see its CMakeLists.txt).
if(NOT WIN32)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/nxai-utilities)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/nxai-utilities/include)
endif()

# Add C Postprocessor project
# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-c-example)

# Add C Image Postprocessor project
# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-c-image-example)

# Add C Raw Postprocessor project
# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-c-raw-example)

# Add Postprocessor Python project
# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-python-example)

# Add Image Postprocessor Python project
# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-python-image-example)

# Add Confidence Postprocessor Python project
# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-python-confidences-example)

# Add Settings Postprocessor Python project
# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-python-settings-example)

# Add Postprocessor Python Events project
# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-python-events-example)

# Add Postprocessor Python NoResponse project
# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-python-noresponse-example)

# Add Cloud Inference Postprocessor Python project
# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-cloud-inference-example)

# Add Geoposition Postprocessor Python project
# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-python-geoposition)

# Add Geoposition Postprocessor Python project
# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-python-geoposition)

# Add Preprocessor Python project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-python-pass-event)

# Add CLIP Situation Postprocessor Python project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-python-clip-situation)

# Add installation option
install(TARGETS
#     postprocessor-c-example
#     postprocessor-c-image-example
#     postprocessor-c-raw-example
     DESTINATION /opt/networkoptix-metavms/mediaserver/bin/plugins/nxai_plugin/nxai_manager/postprocessors/
)

# Install postprocessors
install(PROGRAMS
    # ${CMAKE_CURRENT_BINARY_DIR}/postprocessor-python-example/postprocessor-python-example
    # ${CMAKE_CURRENT_BINARY_DIR}/postprocessor-python-image-example/postprocessor-python-image-example
    # ${CMAKE_CURRENT_BINARY_DIR}/postprocessor-python-confidences-example/postprocessor-python-confidences-example
    # ${CMAKE_CURRENT_BINARY_DIR}/postprocessor-python-settings-example/postprocessor-python-settings-example
    # ${CMAKE_CURRENT_BINARY_DIR}/preprocessor-python-example/preprocessor-python-example
    # ${CMAKE_CURRENT_BINARY_DIR}/postprocessor-python-noresponse-example/postprocessor-python-noresponse-example
    # ${CMAKE_CURRENT_BINARY_DIR}/postprocessor-python-noresponse-example/postprocessor-python-geoposition
    # ${CMAKE_CURRENT_BINARY_DIR}/postprocessor-python-events-example/postprocessor-python-events-example
    # ${CMAKE_CURRENT_BINARY_DIR}/postprocessor-cloud-inference-example/postprocessor-cloud-inference-example
    DESTINATION /opt/networkoptix-metavms/mediaserver/bin/plugins/nxai_plugin/nxai_manager/postprocessors/
)

# postprocessor-python-clip-situation is built with PyInstaller --onedir (see that
# component's CMakeLists.txt for why: --onefile's self-extraction cost the AI
# Manager's short IPC-readiness window, so frames never got routed to it). Its
# artifact is a DIRECTORY holding the exe plus _internal/. Install its *contents*
# flat into postprocessors/ so that <exe dir>/../etc and
# <exe dir>/../nxai-utilities-v5 still resolve the same way they did under --onefile.
install(DIRECTORY
    ${CMAKE_CURRENT_BINARY_DIR}/postprocessor-python-clip-situation/postprocessor-python-clip-situation/
    DESTINATION /opt/networkoptix-metavms/mediaserver/bin/plugins/nxai_plugin/nxai_manager/postprocessors/
    USE_SOURCE_PERMISSIONS
)

# Install preprocessors
# install(PROGRAMS
#     ${CMAKE_CURRENT_BINARY_DIR}/preprocessor-python-example/preprocessor-python-example
#     DESTINATION /opt/networkoptix-metavms/mediaserver/bin/plugins/nxai_plugin/nxai_manager/preprocessors/
# )