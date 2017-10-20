cmake_minimum_required(VERSION 3.9.4)
project(carddetector)

if (USE_OCR)
    add_definitions(-DUSE_OCR)
endif ()

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()
find_package(PkgConfig)

# build flags
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# boost
find_package(Boost COMPONENTS filesystem REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

# opencv
find_package(OpenCV REQUIRED)

# gtk
pkg_check_modules(GTKMM gtkmm-2.4) # look into FindPkgConfig.cmake,
link_directories(${GTKMM_LIBRARY_DIRS})

include_directories(${GTKMM_INCLUDE_DIRS})

# sfml
#find_package(SFML COMPONENTS window graphics system REQUIRED )
#include_directories(${SFML_INCLUDE_DIR})

# general
include_directories(/usr/local/include/)
link_directories(/usr/local/lib/)

# application
set(SOURCE_FILES main.cpp)
add_executable(carddetector ${SOURCE_FILES})
target_link_libraries(carddetector
        ${OpenCV_LIBS}
        ${SFML_LIBRARIES}
        ${Boost_LIBRARIES}
        ${GTKMM_LIBRARIES}
        )
if (USE_OCR)
    target_link_libraries(carddetector tesseract)
endif ()
