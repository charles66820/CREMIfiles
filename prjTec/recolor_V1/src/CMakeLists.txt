# Minimal require version of CMake needed
cmake_minimum_required (VERSION 2.6)

option(ENABLE_DEBUG ON)

# The project name followed by the programming language use
project (recolor_text C)

# for the compilation, -std=c99 to compile under the standard c99, -g for the debuggage, -Wall to show compilation bug, --coverage to check if all the test go in all the code
if(ENABLE_DEBUG)
    set(CMAKE_C_FLAGS "-g -Wall -std=c99 --coverage") # mode debug
else()
    set(CMAKE_C_FLAGS "-O3 -std=c99") # mode -03 to optimise the compilation
endif()

include(CTest)
enable_testing()

### Implementation of all executable
# Creation of the library game
add_library (game game.c game_io.c)
target_link_libraries(game m)

# executable of the game
add_executable (recolor_text recolor_text.c)
target_link_libraries (recolor_text game)

# executable of the bug finder
add_executable (test_game_arblondeau test_game_arblondeau.c)
target_link_libraries (test_game_arblondeau game)

add_executable (test_game_cgoedefroit test_game_cgoedefroit.c)
target_link_libraries (test_game_cgoedefroit game)

add_executable (test_game_vandrault test_game_vandrault.c)
target_link_libraries (test_game_vandrault game)

# Those line launch the test to find the bug
add_test(test_vandrault test_game_vandrault new)
add_test(test_vandrault test_game_vandrault new_empty)
add_test(test_vandrault test_game_vandrault set_cell_init)
add_test(test_vandrault test_game_vandrault set_max_moves)

add_test(test_arblondeau test_game_arblondeau game_nb_moves_max)
add_test(test_arblondeau test_game_arblondeau game_nb_moves_cur)
add_test(test_arblondeau test_game_arblondeau game_cell_current_color)
add_test(test_arblondeau test_game_arblondeau game_play_one_move)

add_test(test_cgoedefroit test_game_cgoedefroit copy)
add_test(test_cgoedefroit test_game_cgoedefroit delete)
add_test(test_cgoedefroit test_game_cgoedefroit is_over)
add_test(test_cgoedefroit test_game_cgoedefroit restart)

set_tests_properties(test_vandrault test_arblondeau test_cgoedefroit PROPERTIES TIMEOUT 10)
