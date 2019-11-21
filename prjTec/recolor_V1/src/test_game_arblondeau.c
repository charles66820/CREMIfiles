#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "game.h"

/**
 * First of all, it will try if the game g created work properly
 * and after it will try if the nbMax chosen is the one implemented,
 * if not it return an error case.
 **/
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
 * First of all, it will try if the game g created work properly
 * and after it will try if the nb_moves_cur match the real one,
 * if not it returns an error case.
 * It will test if : 'nb_moves_cur does not start at 0'
 *                   'Does not increment properly (it suppose to add 1 not anything else'
 *                   'Is he over nbMax'
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
 * First of all, it will try if the game g created work properly
 * If the cells colors does not match the one initialize it return an error case.
 * And if the cells does not change properly after one move it return an error case.
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
 * First of all, it will try if the game g created work properly
 * We also create a copy of the game, and test if it's properly created
 * After, it will test if the game and the copy are the same, if not it return an error case.
 * The game need to change after one move, so if the copy and the game are still the same it return an error case.
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
 * The function main, will take the args to try each function previously created
 * It return an error case if no test was found (like by putting a false args or by not putting one)
 * else it try the args and return SUCCES if no bug was found or FAILURE + the good error case if a bug was found.
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