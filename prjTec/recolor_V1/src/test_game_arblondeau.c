#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "game.h"

/**
 * @brief test if nb_moves_max is properly created
 *
 * @param cells tab of colors
 * @param nb_moves_max number of max hit
 * @param g is test if NULL
 * @param nbMax is test if equal to nb_moves_max
 * @return false + an error message in case of error
 * @return true if no bug was found
 */
bool test_game_nb_moves_max() {
  uint nbMax = 12;
  color cells[SIZE * SIZE] = {
      0, 0, 0, 2, 0, 2, 1, 0, 1, 0, 3, 0, 0, 3, 3, 1, 1, 1, 1, 3, 2, 0, 1, 0,
      1, 0, 1, 2, 3, 2, 3, 2, 0, 3, 3, 2, 2, 3, 1, 0, 3, 2, 1, 1, 1, 2, 2, 0,
      2, 1, 2, 3, 3, 3, 3, 2, 0, 1, 0, 0, 0, 3, 3, 0, 1, 1, 2, 3, 3, 2, 1, 3,
      1, 1, 2, 2, 2, 0, 0, 1, 3, 1, 1, 2, 1, 3, 1, 3, 1, 0, 1, 0, 1, 3, 3, 3,
      0, 3, 0, 1, 0, 0, 2, 1, 1, 1, 3, 0, 1, 3, 1, 0, 0, 0, 3, 2, 3, 1, 0, 0,
      1, 3, 3, 1, 1, 2, 2, 3, 2, 0, 0, 2, 2, 0, 2, 3, 0, 1, 1, 1, 2, 3, 0, 1};
  game g = game_new(cells, nbMax);
  if (g == NULL) {
    fprintf(stderr, "Error : invalid new game \n");
    game_delete(g);
    return false;
  }

  if (game_nb_moves_max(g) != nbMax) {
    fprintf(stderr, "Error : nbMaxMoves is not equal to nbMax.\n");
    game_delete(g);
    return false;
  }
  game_delete(g);
  return true;
}

/**
 * @brief test if nb_moves_cur start at 0 and increment 1 by 1
 *
 * @param cells tab of colors
 * @param nb_moves_max number of max hit
 * @param g is test if NULL
 * @param nb_moves_cur is test if it start at 0 or not, and test if it increment properly
 * @return false + an error message in case of error
 * @return true if no bug was found
 */
bool test_game_nb_moves_cur() {
  uint nbMax = 12;
  color cells[SIZE * SIZE] = {
      0, 0, 0, 2, 0, 2, 1, 0, 1, 0, 3, 0, 0, 3, 3, 1, 1, 1, 1, 3, 2, 0, 1, 0,
      1, 0, 1, 2, 3, 2, 3, 2, 0, 3, 3, 2, 2, 3, 1, 0, 3, 2, 1, 1, 1, 2, 2, 0,
      2, 1, 2, 3, 3, 3, 3, 2, 0, 1, 0, 0, 0, 3, 3, 0, 1, 1, 2, 3, 3, 2, 1, 3,
      1, 1, 2, 2, 2, 0, 0, 1, 3, 1, 1, 2, 1, 3, 1, 3, 1, 0, 1, 0, 1, 3, 3, 3,
      0, 3, 0, 1, 0, 0, 2, 1, 1, 1, 3, 0, 1, 3, 1, 0, 0, 0, 3, 2, 3, 1, 0, 0,
      1, 3, 3, 1, 1, 2, 2, 3, 2, 0, 0, 2, 2, 0, 2, 3, 0, 1, 1, 1, 2, 3, 0, 1};
  game g = game_new(cells, nbMax);
  if (g == NULL) {
    fprintf(stderr, "Error : invalid new game \n");
    game_delete(g);
    return false;
  }

  if (game_nb_moves_cur(g) != 0) {
    fprintf(stderr, "Error :  nbCurrentMoves did not start at 0.\n");
    game_delete(g);
    return false;
  }

  game_play_one_move(g, BLUE);
  if (game_nb_moves_cur(g) != 1) {
    fprintf(stderr, "Error : NbCurrentMoves does not increment properly.\n");
    game_delete(g);
    return false;
  }

  if (game_nb_moves_cur(g) > nbMax) {
    fprintf(stderr, "Error : nbCurrentMoves is over to nbMax.\n");
    game_delete(g);
    return false;
  }
  game_delete(g);
  return true;
}

/**
 * @brief test if the cells are properly created, and change properly
 *
 * @param cells tab of colors
 * @param nb_moves_max number of max hit
 * @param g is test if NULL
 * @param cell_current_color is test if it was properly created and change properly
 * @return false + an error message in case of error
 * @return true if no bug was found
 */
bool test_game_cell_current_color() {
  uint nbMax = 12;
  color cells[SIZE * SIZE] = {
      0, 0, 0, 2, 0, 2, 1, 0, 1, 0, 3, 0, 0, 3, 3, 1, 1, 1, 1, 3, 2, 0, 1, 0,
      1, 0, 1, 2, 3, 2, 3, 2, 0, 3, 3, 2, 2, 3, 1, 0, 3, 2, 1, 1, 1, 2, 2, 0,
      2, 1, 2, 3, 3, 3, 3, 2, 0, 1, 0, 0, 0, 3, 3, 0, 1, 1, 2, 3, 3, 2, 1, 3,
      1, 1, 2, 2, 2, 0, 0, 1, 3, 1, 1, 2, 1, 3, 1, 3, 1, 0, 1, 0, 1, 3, 3, 3,
      0, 3, 0, 1, 0, 0, 2, 1, 1, 1, 3, 0, 1, 3, 1, 0, 0, 0, 3, 2, 3, 1, 0, 0,
      1, 3, 3, 1, 1, 2, 2, 3, 2, 0, 0, 2, 2, 0, 2, 3, 0, 1, 1, 1, 2, 3, 0, 1};
  game g = game_new(cells, nbMax);
  if (g == NULL) {
    fprintf(stderr, "Error : invalid new game \n");
    game_delete(g);
    return false;
  }

  for (int y = 0; y < SIZE; y++) {
    for (int x = 0; x < SIZE; x++) {
      if (game_cell_current_color(g, x, y) != cells[x + SIZE * y]) {
        fprintf(stderr,
                "Error: game cells are not equal to initial game cells.\n");
        game_delete(g);
        return false;
      }
    }
  }

  game_play_one_move(g, BLUE);
  if (game_cell_current_color(g, 0, 0) != BLUE) {
    fprintf(stderr, "Error : The game does not change the color.\n");
    game_delete(g);
    return false;
  }

  game_delete(g);
  return true;
}

/**
 * @brief test if the play change the game properly
 *
 * @param cells tab of colors
 * @param nb_moves_max number of max hit
 * @param g is test if NULL
 * @param gc is test if NULL
 * @return false + an error message in case of error
 * @return true if no bug was found
 */
bool test_game_play_one_move() {
  uint nbMax = 12;
  color cells[SIZE * SIZE] = {
      0, 0, 0, 2, 0, 2, 1, 0, 1, 0, 3, 0, 0, 3, 3, 1, 1, 1, 1, 3, 2, 0, 1, 0,
      1, 0, 1, 2, 3, 2, 3, 2, 0, 3, 3, 2, 2, 3, 1, 0, 3, 2, 1, 1, 1, 2, 2, 0,
      2, 1, 2, 3, 3, 3, 3, 2, 0, 1, 0, 0, 0, 3, 3, 0, 1, 1, 2, 3, 3, 2, 1, 3,
      1, 1, 2, 2, 2, 0, 0, 1, 3, 1, 1, 2, 1, 3, 1, 3, 1, 0, 1, 0, 1, 3, 3, 3,
      0, 3, 0, 1, 0, 0, 2, 1, 1, 1, 3, 0, 1, 3, 1, 0, 0, 0, 3, 2, 3, 1, 0, 0,
      1, 3, 3, 1, 1, 2, 2, 3, 2, 0, 0, 2, 2, 0, 2, 3, 0, 1, 1, 1, 2, 3, 0, 1};
  game g = game_new(cells, nbMax);
  game gc = game_copy(g);
  if (g == NULL) {
    fprintf(stderr, "Error : invalid new game \n");
    game_delete(g);
    return false;
  }
  if (gc == NULL) {
    fprintf(stderr, "Error : invalid new game copy \n");
    game_delete(gc);
    return false;
  }

  for (int y = 0; y < SIZE; y++) {
    for (int x = 0; x < SIZE; x++) {
      if (game_cell_current_color(gc, x, y) !=
          game_cell_current_color(g, x, y)) {
        fprintf(stderr, "Error: game and copy game cells are not equal!\n");
        game_delete(g);
        game_delete(gc);
        return false;
      }
    }
  }

  game_play_one_move(g, BLUE);
  if (game_cell_current_color(g, 0, 0) != BLUE) {
    fprintf(stderr, "Error : game_play_one_move does not change the game.\n");
    game_delete(g);
    game_delete(gc);
    return false;
  }
  game_delete(g);
  game_delete(gc);
  return true;
}

/**
 * @brief main fuction will call other functions to test bugs
 *
 * @param argv[1] wait for one of the fourth good parameter
 * @return EXIT_FAILURE + an error message in case of error
 * @return EXIT_SUCCESS if no bug was found
 */
int main(int argc, char const *argv[]) {
  if (argc == 1) {
    fprintf(stderr, "Usage: %s <testname> [<...>]\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  bool ok = false;

  if (!strcmp(argv[1], "game_nb_moves_max"))
    ok = test_game_nb_moves_max();
  else if (!strcmp(argv[1], "game_nb_moves_cur"))
    ok = test_game_nb_moves_cur();
  else if (!strcmp(argv[1], "game_cell_current_color"))
    ok = test_game_cell_current_color();
  else if (!strcmp(argv[1], "game_play_one_move"))
    ok = test_game_play_one_move();
  else {
    fprintf(stderr, "Error: test \"%s\" not found!\n", argv[1]);
    exit(EXIT_FAILURE);
  }
  if (ok) {
    fprintf(stderr, "Test \"%s\" finished: SUCCESS\n", argv[1]);
    return EXIT_SUCCESS;
  } else {
    fprintf(stderr, "Test \"%s\" finished: FAILURE\n", argv[1]);
    return EXIT_FAILURE;
  }
}