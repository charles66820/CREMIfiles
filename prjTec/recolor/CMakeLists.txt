# Minimal require version of CMake needed
cmake_minimum_required (VERSION 2.6)

option(ENABLE_DEBUG "debug options" ON)

# The project name followed by the programming language use
project("Recolor" LANGUAGES C)

# Compile all the files without any particular specification
set(CMAKE_C_FLAGS "-std=c99 -Wall")
# Compile all the files in a DEBUG mode (add debug and coverage)
set(CMAKE_C_FLAGS_DEBUG "-g -DDEBUG --coverage")
# Compile all the file on the release mode (optimise version of the compilation faster than the normal version)
set(CMAKE_C_FLAGS_RELEASE "-O3")

# for SDL2
IF(WIN32)
    set(SDL2_ALL_LIBS mingw32 SDL2main SDL2 SDL2_image SDL2_ttf)
    set(SDL2_ALL_INC C:/TDM-GCC-32/include/SDL2)

ELSE()
    # pkg-config --list-all  | grep -i sdl2
    include(FindPkgConfig)
    pkg_search_module(SDL2 REQUIRED sdl2)
    pkg_search_module(SDL2_IMG REQUIRED SDL2_image)
    pkg_search_module(SDL2_TTF REQUIRED SDL2_ttf)
    set(SDL2_ALL_LIBS ${SDL2_LIBRARIES} ${SDL2_IMG_LIBRARIES} ${SDL2_TTF_LIBRARIES})
    set(SDL2_ALL_INC ${SDL2_INCLUDE_DIRS} ${SDL2_IMG_INCLUDE_DIRS} ${SDL2_TTF_INCLUDE_DIRS})
ENDIF()

include_directories(${SDL2_ALL_INC} include)

# for generate bin and lib files in generated folder
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "generated")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "generated")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "generated")
set(EXECUTABLE_OUTPUT_PATH "generated")

# copy data and assets folders to generated folder
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/assets DESTINATION ${EXECUTABLE_OUTPUT_PATH})

# uninstall target
if(NOT TARGET uninstall)
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)

  add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()

### Implementation of all executable
# Creation of the library game
add_library (game lib/game.c lib/game_io.c lib/game_rand.c)
IF (WIN32)
  add_library (gl lib/getline.c)
  target_link_libraries(game gl)
ENDIF()

add_library (sol lib/solution.c)

# executable of the game
add_executable (recolor_text bin/recolor_text.c)
target_link_libraries (recolor_text game)

add_executable(recolor_solve bin/recolor_solve.c)
target_link_libraries (recolor_solve pthread game m sol)

IF(WIN32)
    add_executable(recolor_sdl WIN32 bin/main_sdl.c bin/recolor_sdl.c assets/ressourses.rc)
    target_compile_options(recolor_sdl INTERFACE "-mwindows")
ELSE()
    add_executable(recolor_sdl bin/main_sdl.c bin/recolor_sdl.c assets/ressourses.rc)
ENDIF()
target_link_libraries(recolor_sdl ${SDL2_ALL_LIBS} game m)

# executable of the bug finder
add_executable (test_game_arblondeau tests/test_game_arblondeau.c)
target_link_libraries (test_game_arblondeau game)

add_executable (test_game_cgoedefroit tests/test_game_cgoedefroit.c)
target_link_libraries (test_game_cgoedefroit game)

add_executable (test_game_vandrault tests/test_game_vandrault.c)
target_link_libraries (test_game_vandrault game)

add_executable(test_solution_arblondeau tests/test_solution_arblondeau.c)
target_link_libraries (test_solution_arblondeau game sol)

### Install
install(TARGETS recolor_text recolor_solve recolor_sdl CONFIGURATIONS Release RUNTIME DESTINATION bin)
install(DIRECTORY assets/ DESTINATION bin/assets)

### Uninstall

### Tests
include(CTest)
enable_testing()

# Those line launch the test to find the bug
add_test(test_vandrault_new generated/test_game_vandrault new)
add_test(test_vandrault_new_empty generated/test_game_vandrault new_empty)
add_test(test_vandrault_set_cell_init generated/test_game_vandrault set_cell_init)
add_test(test_vandrault_set_max_moves generated/test_game_vandrault set_max_moves)
add_test(test_vandrault_height generated/test_game_vandrault height)
add_test(test_vandrault_wrapping generated/test_game_vandrault wrapping)

add_test(test_arblondeau_nb_moves_max generated/test_game_arblondeau game_nb_moves_max)
add_test(test_arblondeau_nv_moves_cur generated/test_game_arblondeau game_nb_moves_cur)
add_test(test_arblondeau_cell_current_color generated/test_game_arblondeau game_cell_current_color)
add_test(test_arblondeau_play_one_move generated/test_game_arblondeau game_play_one_move)
add_test(test_arblondeau_width generated/test_game_arblondeau game_width)
add_test(test_arblondeau_new_empty_ext generated/test_game_arblondeau game_new_empty_ext)
add_test(test_arblondeau_create_solution generated/test_solution_arblondeau create_solution)
add_test(test_arblondeau_len_solution generated/test_solution_arblondeau len_solution)
add_test(test_arblondeau_string_solution generated/test_solution_arblondeau string_solution)
add_test(test_arblondeau_delete_solution generated/test_solution_arblondeau delete_solution)

add_test(test_cgoedefroit_copy generated/test_game_cgoedefroit copy)
add_test(test_cgoedefroit_delete generated/test_game_cgoedefroit delete)
add_test(test_cgoedefroit_is_over generated/test_game_cgoedefroit is_over)
add_test(test_cgoedefroit_restart generated/test_game_cgoedefroit restart)
add_test(test_cgoedefroit_new_ext generated/test_game_cgoedefroit new_ext)
add_test(test_cgoedefroit_save generated/test_game_cgoedefroit save)
add_test(test_cgoedefroit_load generated/test_game_cgoedefroit load)
add_test(test_cgoedefroit_save2load generated/test_game_cgoedefroit save2load)
add_test(test_cgoedefroit_load2save generated/test_game_cgoedefroit load2save)

set_tests_properties(test_vandrault_new test_vandrault_new_empty test_vandrault_set_cell_init test_vandrault_set_max_moves test_vandrault_height test_vandrault_wrapping test_arblondeau_nb_moves_max test_arblondeau_nv_moves_cur test_arblondeau_cell_current_color test_arblondeau_play_one_move test_arblondeau_width test_arblondeau_new_empty_ext test_cgoedefroit_copy test_cgoedefroit_delete test_cgoedefroit_is_over test_cgoedefroit_restart test_cgoedefroit_new_ext test_cgoedefroit_save test_cgoedefroit_load test_cgoedefroit_save2load test_cgoedefroit_load2save test_arblondeau_create_solution test_arblondeau_len_solution test_arblondeau_string_solution test_arblondeau_delete_solution PROPERTIES TIMEOUT 10 PARALLEL_LEVEL 8)
