#C'est la version minimal du cmake à utilisé
cmake_minimum_required (VERSION 2.6)

#C'est le nom du projet et le langage de programmation utilisé
project (recolor_text C)

#Pour la compilation, -std=c99 pour compiler sous la norme c99, -g pour le debuggage, -Wall pour afficher les erreurs de compilation
set (CMAKE_C_FLAGS "-std=c99 -g -Wall")

enable_testing()

#Implémentation des executables
#   Création de la librairie game
add_library (game game.c game_io.c) 
target_link_libraries(game m)

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

#Permet de lancer les tests
add_test(recolor_text recolor_text)

add_test(test_arblondeau test_game_arblondeau)

add_test(test_vandrault test_game_vandrault)

add_test(test_cgoedefroit test_game_cgoedefroit)