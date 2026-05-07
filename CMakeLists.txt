cmake_minimum_required(VERSION 3.14)

project(vggtcppinfer LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 75)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

find_package(CUDAToolkit 12.8 REQUIRED)

set(Torch_ROOT "D:/libtorch/Release/libtorch")
set(Torch_DIR "${Torch_ROOT}/share/cmake/Torch")
find_package(Torch REQUIRED)

set(TensorRT_DIR "D:/TensorRT/TensorRT-10.16.1.11")

include_directories(
    ${TensorRT_DIR}/include
)

find_library(TENSORRT_NVINFER_LIB
    NAMES nvinfer_10
    PATHS ${TensorRT_DIR}/lib
    REQUIRED
)

find_library(TENSORRT_NVONNXPARSER_LIB
    NAMES nvonnxparser_10
    PATHS ${TensorRT_DIR}/lib
    REQUIRED
)

find_library(TENSORRT_PLUGIN_LIB
    NAMES nvinfer_plugin_10
    PATHS ${TensorRT_DIR}/lib
    REQUIRED
)

add_executable(vggt_cpp_infer
    test.cpp
    OnnxVGGT.h
)

target_link_libraries(vggt_cpp_infer 
    "${TORCH_LIBRARIES}"
    ${TENSORRT_NVINFER_LIB}
    ${TENSORRT_NVONNXPARSER_LIB}
    ${TENSORRT_PLUGIN_LIB}
)

#[[
if (WIN32)
    file(GLOB TORCH_DLLS "${Torch_ROOT}/lib/*.dll")

    add_custom_command(TARGET vggt_cpp_infer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:vggt_cpp_infer>
        COMMENT "Copying libtorch DLLs to output directory\n"
    )
endif()


if (WIN32)
    file(GLOB TensorRT_DLLS "${TensorRT_DIR}/bin/*.dll")

    add_custom_command(TARGET vggt_cpp_infer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TensorRT_DLLS}
            $<TARGET_FILE_DIR:vggt_cpp_infer>
        COMMENT "Copying tensorRT DLLs to output directory\n"
    )
endif()
]]