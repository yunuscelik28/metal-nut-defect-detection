cmake_minimum_required(VERSION 3.28.3)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(Vision_Project)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Release by default, a whole folder (220 images) runs a lot faster with -O3
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED)

add_executable(mtest src/main.cpp src/image_processor.cpp src/defect_detector.cpp)
target_include_directories(mtest PRIVATE
    ${CMAKE_SOURCE_DIR}/include/vision
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(mtest PRIVATE ${OpenCV_LIBS})

# Show warnings (signed/unsigned compare, unused variables...)
if(NOT MSVC)
    target_compile_options(mtest PRIVATE -Wall -Wextra)
endif()
