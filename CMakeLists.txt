cmake_minimum_required(VERSION 3.28.3)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(Vision_Project)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} )

add_executable(mtest src/main.cpp src/image_processor.cpp src/defect_detector.cpp)
target_link_libraries(mtest ${OpenCV_LIBS})
target_include_directories(mtest PRIVATE
    ${CMAKE_SOURCE_DIR}/include/vision
)
