cmake_minimum_required(VERSION 3.8)
project(carddetector)

if (USE_OCR)
    add_definitions(-DUSE_OCR)
endif ()

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()


set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
include_directories(/usr/local/include/)
link_directories(/usr/local/lib/)

set(SOURCE_FILES main.cpp)
add_executable(carddetector ${SOURCE_FILES})

target_link_libraries(carddetector ${OpenCV_LIBS})

if (USE_OCR)
    target_link_libraries(carddetector tesseract)
endif ()
