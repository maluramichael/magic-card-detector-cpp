cmake_minimum_required(VERSION 3.9.4)
project(carddetector)

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

# general
include_directories(/usr/local/include/)
link_directories(/usr/local/lib/)

# application
add_executable(carddetector main.cpp)
add_executable(hash Hash.cpp)
add_executable(ocr OCR.cpp)
target_link_libraries(carddetector ${OpenCV_LIBS} ${Boost_LIBRARIES} tesseract)
target_link_libraries(hash ${OpenCV_LIBS} ${Boost_LIBRARIES} tesseract)
target_link_libraries(ocr ${OpenCV_LIBS} ${Boost_LIBRARIES} tesseract)
