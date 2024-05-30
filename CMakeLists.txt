cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(ImageProcessing LANGUAGES CXX CUDA)

cmake_policy(SET CMP0115 NEW)
cmake_policy(SET CMP0104 NEW)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_RUNTIME_LIBRARY "Shared")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-as-needed")
set(CMAKE_C_FLAGS_PROFILE "-pg -g -O2" CACHE STRING "")                          

set(ENV{PATH} "/usr/local/cuda/bin:$ENV{PATH}")
set(ENV{LD_LIBRARY_PATH} "/usr/local/cuda/lib64:$ENV{LD_LIBRARY_PATH}")


find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED) 

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

add_executable(image_processing 
    src/main.cpp
    src/CudaImageProcessor.cu
)

set_target_properties(image_processing PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75"
)

target_link_libraries(image_processing PRIVATE ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CUDA_cudart_LIBRARY} ${CMAKE_DL_LIBS} pthread rt)
