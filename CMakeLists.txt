cmake_minimum_required(VERSION 3.10)
project(ImageProcessing)

cmake_policy(SET CMP0115 NEW)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

add_executable(image_processing 
    src/main.cpp
    src/image_processor.cpp
)

target_link_libraries(image_processing ${OpenCV_LIBS} ${CUDA_LIBRARIES})
