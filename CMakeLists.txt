cmake_minimum_required(VERSION 3.13)

project(WiiUPark LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CREMA_DIR "${CMAKE_SOURCE_DIR}/third_party/wiiu-crema")

if(NOT EXISTS "${CREMA_DIR}/crema/crema_app.c")
    message(FATAL_ERROR "Crema framework was not downloaded.")
endif()

add_library(crema STATIC
    ${CREMA_DIR}/crema/crema_app.c
    ${CREMA_DIR}/crema/crema_audio.c
    ${CREMA_DIR}/crema/crema_bank.c
    ${CREMA_DIR}/crema/crema_blend.c
    ${CREMA_DIR}/crema/crema_buffer.c
    ${CREMA_DIR}/crema/crema_effect.c
    ${CREMA_DIR}/crema/crema_frame.c
    ${CREMA_DIR}/crema/crema_hud.c
    ${CREMA_DIR}/crema/crema_collide.c
    ${CREMA_DIR}/crema/crema_entity.c
    ${CREMA_DIR}/crema/crema_input.c
    ${CREMA_DIR}/crema/crema_mesh.c
    ${CREMA_DIR}/crema/crema_music.c
    ${CREMA_DIR}/crema/crema_pak.c
    ${CREMA_DIR}/crema/crema_save.c
    ${CREMA_DIR}/crema/crema_scene.c
    ${CREMA_DIR}/crema/crema_shader.cpp
    ${CREMA_DIR}/crema/crema_texture.c
)

target_include_directories(crema PUBLIC
    "${CREMA_DIR}/crema"
    "${CREMA_DIR}/external"
)

add_executable(WiiUPark
    source/main.cpp
)

target_link_libraries(WiiUPark PRIVATE crema)

wut_create_rpx(WiiUPark)

wut_create_wuhb(WiiUPark
    NAME "The Park"
    SHORTNAME "ThePark"
    AUTHOR "Wii U Homebrew"
)
