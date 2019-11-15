# C'est la version minimal du cmake à utilisé
cmake_minimum_required (VERSION 2.6)

option(ENABLE_DEBUG ON)

# C'est le nom du projet et le langage de programmation utilisé
project (recolor_text C)

# Pour la compilation, -std=c99 pour compiler sous la norme c99, -g pour le debuggage, -Wall pour afficher les erreurs de compilation
if(ENABLE_DEBUG)
    set(CMAKE_C_FLAGS "-g -Wall -std=c99")
else()
    set(CMAKE_C_FLAGS "-O3 -std=c99") # -03 pour optimier la compilation
endif()


enable_testing()

# Implémentation des executables
#   Création de la librairie game
add_library (game game.c game_io.c)
target_link_libraries (game m)

#   executable du jeu
add_executable (recolor_text recolor_text.c)
target_link_libraries (recolor_text game)

#   executable des tests de bug
add_executable (test_game_arblondeau test_game_arblondeau.c)
target_link_libraries (test_game_arblondeau game)

add_executable (test_game_cgoedefroit test_game_cgoedefroit.c)
target_link_libraries (test_game_cgoedefroit game)

add_executable (test_game_vandrault test_game_vandrault.c)
target_link_libraries (test_game_vandrault game)

# Permet de lancer les tests
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
