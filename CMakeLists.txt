cmake_minimum_required(VERSION 3.10)
project(ImageProcessing)

set(CMAKE_CXX_STANDARD 14)
find_package(OpenCV REQUIRED PATHS /usr/include/opencv4 NO_DEFAULT_PATH)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(image_processing src/main.cpp)
target_link_libraries(image_processing ${OpenCV_LIBS})
