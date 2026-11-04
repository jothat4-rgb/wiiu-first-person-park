cmake_minimum_required(VERSION 3.7)

project(WiiUPark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(WiiUPark
    source/main.cpp
)

wut_create_rpx(WiiUPark)

install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/WiiUPark.rpx"
    DESTINATION "${CMAKE_INSTALL_PREFIX}"
)
