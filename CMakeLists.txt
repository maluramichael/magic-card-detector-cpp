cmake_minimum_required(VERSION 3.8)
project(carddetector)

set(CMAKE_CXX_STANDARD 14)

find_package( OpenCV REQUIRED )

set(SOURCE_FILES main.cpp)
add_executable(carddetector ${SOURCE_FILES})
target_link_libraries( carddetector ${OpenCV_LIBS} )
