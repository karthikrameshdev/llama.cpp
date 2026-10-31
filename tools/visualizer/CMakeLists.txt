cmake_minimum_required(VERSION 3.20)
project(llama_visualizer CXX)
set(LLAMA_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." CACHE PATH "llama.cpp checkout")
set(LLAMA_BUILD_DIR "${LLAMA_SOURCE_DIR}/build" CACHE PATH "Existing shared llama.cpp build")
find_package(Threads REQUIRED)
foreach(LIB llama llama-common ggml ggml-base mtmd)
    string(REPLACE "-" "_" VAR ${LIB})
    find_library(${VAR}_LIBRARY NAMES ${LIB} PATHS "${LLAMA_BUILD_DIR}/bin" NO_DEFAULT_PATH REQUIRED)
    list(APPEND OBSERVER_LIBRARIES "${${VAR}_LIBRARY}")
endforeach()
add_executable(llama-visualizer-observer native/observer.cpp)
target_compile_features(llama-visualizer-observer PRIVATE cxx_std_17)
target_include_directories(llama-visualizer-observer PRIVATE
    "${LLAMA_SOURCE_DIR}/include" "${LLAMA_SOURCE_DIR}/ggml/include"
    "${LLAMA_SOURCE_DIR}/common" "${LLAMA_SOURCE_DIR}/vendor/nlohmann"
    "${LLAMA_SOURCE_DIR}/tools/mtmd")
target_link_libraries(llama-visualizer-observer PRIVATE ${OBSERVER_LIBRARIES} Threads::Threads)
set_target_properties(llama-visualizer-observer PROPERTIES BUILD_RPATH "${LLAMA_BUILD_DIR}/bin")
