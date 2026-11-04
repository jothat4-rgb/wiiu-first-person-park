cmake_minimum_required(VERSION 3.7)

project(WiiUPark VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(WiiUPark
    main.cpp
)

wut_create_rpx(WiiUPark)
