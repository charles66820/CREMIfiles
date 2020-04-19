#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "game.h"
#include "game_io.h"

/**
 * @brief Unite test for game_copy
 *
 * @return bool
 */
bool test_game_copy() {
  uint nbMaxHit = 13;

  color cells[SIZE * SIZE] = {
      0, 0, 0, 2, 0, 2, 1, 0, 1, 0, 3, 0, 0, 3, 3, 1, 1, 1, 1, 3, 2, 0, 1, 0,
      1, 0, 1, 2, 3, 2, 3, 2, 0, 3, 3, 2, 2, 3, 1, 0, 3, 2, 1, 1, 1, 2, 2, 0,
      2, 1, 2, 3, 3, 3, 3, 2, 0, 1, 0, 0, 0, 3, 3, 0, 1, 1, 2, 3, 3, 2, 1, 3,
      1, 1, 2, 2, 2, 0, 0, 1, 3, 1, 1, 2, 1, 3, 1, 3, 1, 0, 1, 0, 1, 3, 3, 3,
      0, 3, 0, 1, 0, 0, 2, 1, 1, 1, 3, 0, 1, 3, 1, 0, 0, 0, 3, 2, 3, 1, 0, 0,
      1, 3, 3, 1, 1, 2, 2, 3, 2, 0, 0, 2, 2, 0, 2, 3, 0, 1, 1, 1, 2, 3, 0, 1};

  // create new game
  game g = game_new(cells, nbMaxHit);

  // test if game has been create
  if (g == NULL) {
    fprintf(stderr, "Error: invalid new game!\n");

    return false;
  }

  // test of game_copy()
  game gc = game_copy(g);

  // copy test
  if (gc == NULL) {
    fprintf(stderr, "Error: invalid copy game!\n");

    game_delete(g);

    return false;
  }

  // copy of max move test
  if (game_nb_moves_max(gc) != game_nb_moves_max(g)) {
    fprintf(stderr, "Error: game and copy game max moves are not equal!\n");

    game_delete(g);
    game_delete(gc);

    return false;
  }

  // copy of nb current move test
  if (game_nb_moves_cur(gc) != game_nb_moves_cur(g)) {
    fprintf(stderr,
            "Error: game and copy game nb curent move are not equal!\n");

    game_delete(g);
    game_delete(gc);

    return false;
  }

  // copy of cells test
  for (int y = 0; y < game_height(g); y++)
    for (int x = 0; x < game_width(g); x++)
      if (game_cell_current_color(gc, x, y) !=
          game_cell_current_color(g, x, y)) {
        fprintf(stderr, "Error: game and copy game cells are not equal!\n");

        game_delete(g);
        game_delete(gc);

        return false;
      }

  // test of change table cell
  game_set_cell_init(g, 4, 4, BLUE);

  if (game_cell_current_color(g, 4, 4) != BLUE) {
    fprintf(stderr, "Error: invalid game set cell init!\n");

    game_delete(g);
    game_delete(gc);

    return false;
  }

  // test if cells has been copyed
  if (game_cell_current_color(gc, 4, 4) == game_cell_current_color(g, 4, 4)) {
    fprintf(stderr, "Error: game and copy game cells are not equal!\n");

    game_delete(g);
    game_delete(gc);

    return false;
  }

  game_delete(g);
  game_delete(gc);

  return true;
}

/**
 * @brief Unite test for game_delete
 *
 * @return bool
 */
bool test_game_delete() {
  // create new game
  game g = game_new_empty();
  if (g == NULL) {
    fprintf(stderr, "Error: invalid new empty game!\n");
    return false;
  }

  // delete game
  game_delete(g);

  return true;
}

/**
 * @brief Unite test for game_is_over
 *
 * @return bool
 */
bool test_game_is_over() {
  uint nbMaxHit = 12;

  color cells[SIZE * SIZE] = {
      0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0,
      1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0,
      1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0,
      0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
      1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1};

  // create new game
  game g = game_new(cells, nbMaxHit);

  // test if game has been create
  if (g == NULL) {
    fprintf(stderr, "Error: invalid new game!\n");

    return false;
  }

  // test game is over if game is not over
  if (game_is_over(g) == true) {
    fprintf(stderr, "Error: game is over when shouldn't be!\n");

    game_delete(g);

    return false;
  }

  // play 0
  game_play_one_move(g, 0);
  // play 1
  game_play_one_move(g, 1);
  // play 0
  game_play_one_move(g, 0);
  // play 1
  game_play_one_move(g, 1);

  // test game is not over if game is over
  if (game_is_over(g) == false) {
    fprintf(stderr, "Error: game is not over when it should be!\n");

    game_delete(g);

    return false;
  }

  game_delete(g);

  return true;
}

/**
 * @brief Unite test for game_restart
 *
 * @return bool
 */
bool test_game_restart() {
  uint nbMaxHit = 12;

  color cells[SIZE * SIZE] = {
      0, 0, 0, 2, 0, 2, 1, 0, 1, 0, 3, 0, 0, 3, 3, 1, 1, 1, 1, 3, 2, 0, 1, 0,
      1, 0, 1, 2, 3, 2, 3, 2, 0, 3, 3, 2, 2, 3, 1, 0, 3, 2, 1, 1, 1, 2, 2, 0,
      2, 1, 2, 3, 3, 3, 3, 2, 0, 1, 0, 0, 0, 3, 3, 0, 1, 1, 2, 3, 3, 2, 1, 3,
      1, 1, 2, 2, 2, 0, 0, 1, 3, 1, 1, 2, 1, 3, 1, 3, 1, 0, 1, 0, 1, 3, 3, 3,
      0, 3, 0, 1, 0, 0, 2, 1, 1, 1, 3, 0, 1, 3, 1, 0, 0, 0, 3, 2, 3, 1, 0, 0,
      1, 3, 3, 1, 1, 2, 2, 3, 2, 0, 0, 2, 2, 0, 2, 3, 0, 1, 1, 1, 2, 3, 0, 1};

  // create new game
  game g = game_new(cells, nbMaxHit);

  // test if game has been create
  if (g == NULL) {
    fprintf(stderr, "Error: invalid new game!\n");

    return false;
  }

  // change cells

  // play 1
  game_play_one_move(g, GREEN);
  // play 2
  game_play_one_move(g, BLUE);

  // test restart
  game_restart(g);

  // test if game cells is equal to initial game cells
  for (int y = 0; y < game_height(g); y++)
    for (int x = 0; x < game_width(g); x++)
      if (game_cell_current_color(g, x, y) != cells[x + SIZE * y]) {
        fprintf(stderr,
                "Error: game cells are not equal to initial game cells!\n");

        game_delete(g);

        return false;
      }

  // test if current moves is not reset
  if (game_nb_moves_cur(g) != 0) {
    fprintf(stderr, "Error: current moves has not correclly reset!\n");

    game_delete(g);

    return false;
  }

  // test if max moves is not change
  if (game_nb_moves_max(g) != nbMaxHit) {
    fprintf(stderr, "Error: invalid game nb max moves!\n");

    game_delete(g);

    return false;
  }

  game_delete(g);

  return true;
}

/**
 * @brief Unite test for game_new_ext
 *
 * @return bool
 */
bool test_game_new_ext() {
  color cells[4 * 8] = {0, 0, 0, 2, 0, 2, 1, 0, 1, 0, 3, 0, 0, 3, 3, 1,
                        1, 1, 1, 3, 2, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2};
  color cellsw[5 * 7] = {0, 0, 0, 2, 0, 2, 1, 0, 1, 0, 3, 0, 0, 3, 3, 1, 1, 1,
                         1, 3, 2, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 1, 0, 2};

  game g = game_new_ext(4, 8, cells, 11, false);
  // check if game are not create
  if (g == NULL) {
    fprintf(stderr, "Error: invalid new game!\n");
    return false;
  }

  game gw = game_new_ext(5, 7, cellsw, 6, true);

  // check if game are not create
  if (gw == NULL) {
    fprintf(stderr, "Error: invalid new game!\n");
    game_delete(g);
    return false;
  }

  // check if the number of max moves is correctly define
  if (game_nb_moves_max(g) != 11) {
    fprintf(stderr,
            "Error: new game number of max moves is not correctly define!\n");
    game_delete(g);
    game_delete(gw);
    return false;
  }

  // check if game width is correctly define
  if (game_width(g) != 4) {
    fprintf(stderr, "Error: new game width is not correctly define!\n");
    game_delete(g);
    game_delete(gw);
    return false;
  }

  // check if game height is correctly define
  if (game_height(g) != 8) {
    fprintf(stderr, "Error: new game height is not correctly define!\n");
    game_delete(g);
    game_delete(gw);
    return false;
  }

  // check if game wrapping is correctly define
  if (game_is_wrapping(g)) {
    fprintf(stderr, "Error: new game wrapping is not correctly define!\n");
    game_delete(g);
    game_delete(gw);
    return false;
  }

  // check if game wrapping is correctly define
  if (!game_is_wrapping(gw)) {
    fprintf(stderr, "Error: new game wrapping is not correctly define!\n");
    game_delete(g);
    game_delete(gw);
    return false;
  }

  // check if game nb current moves is correctly define
  if (game_nb_moves_cur(g) != 0) {
    fprintf(stderr,
            "Error: new game nb current moves is not correctly define!\n");
    game_delete(g);
    game_delete(gw);
    return false;
  }

  // check if cells is correctly define
  for (uint y = 0; y < 8; y++)
    for (uint x = 0; x < 4; x++)
      if (game_cell_current_color(g, x, y) != cells[x + 4 * y]) {
        fprintf(stderr, "Error: cells of new game is not correctly define!\n");
        game_delete(g);
        game_delete(gw);
        return false;
      }

  for (uint y = 0; y < 7; y++)
    for (uint x = 0; x < 5; x++)
      if (game_cell_current_color(gw, x, y) != cellsw[x + 5 * y]) {
        fprintf(stderr, "Error: cells of new game is not correctly define!\n");
        game_delete(g);
        game_delete(gw);
        return false;
      }

  game_delete(g);
  game_delete(gw);
  return true;
}

/**
 * @brief Unite test for game_save
 *
 * @return bool
 */
bool test_game_save() {
  // create new game
  color cells[4 * 5] = {0, 0, 0, 2, 0, 2, 1, 0, 1, 0,
                        3, 0, 0, 3, 3, 1, 1, 1, 1, 3};
  game g = game_new_ext(5, 4, cells, 7, true);
  char validfilecontent[] =
      "5 4 7 S\n0 0 0 2 0\n2 1 0 1 0\n3 0 0 3 3\n1 1 1 1 3\n";

  // save the game
  game_save(g, "data/savetest.rec");
  game_delete(g);

  // open generated file
  FILE *file;
  file = fopen("data/savetest.rec", "r");
  if (file == NULL) {
    printf("The test file couldn't be open!\n");
    remove("data/savetest.rec");
    return false;
  }

  char filecontent[49];
  if (!fscanf(file, "%49c", filecontent)) {  // load file content
    fprintf(stderr, "Error: unknown!\n");
    // close and remove file
    fclose(file);
    remove("data/savetest.rec");
    return false;
  }
  filecontent[48] = '\0';

  // close and remove file
  fclose(file);
  remove("data/savetest.rec");

  // compare generate file content with valid file content
  if (strcmp(filecontent, validfilecontent)) {
    fprintf(stderr, "Error: the game is not saved correctly in the file!\n");
    return false;
  }

  return true;
}

/**
 * @brief Unite test for game_load
 *
 * @return bool
 */
bool test_game_load() {
  // create data folder
  if (system("mkdir -p data") == -1) {
    fprintf(stderr, "The folder (data) can not be create.\n");
    return false;
  }

  // open new file for test
  FILE *file;
  file = fopen("data/savetest.rec", "w");
  if (file == NULL) {
    printf("The test file couldn't be create!\n");
    return false;
  }

  // fill test file with valid content
  fprintf(file, "5 4 17 S\n0 0 0 2 0\n2 1 0 1 0\n3 0 0 3 3\n1 1 1 1 3\n");
  fclose(file);

  // load file
  game g = game_load("data/savetest.rec");
  if (g == NULL) {
    printf("The game has not been loaded!\n");
    remove("data/savetest.rec");
    return false;
  }

  // remove file
  remove("data/savetest.rec");

  // check if the loaded game is valid
  if (game_width(g) != 5) {
    printf("The width of loaded game are not valid!\n");
    game_delete(g);
    return false;
  }

  if (game_height(g) != 4) {
    printf("The height of loaded game are not valid!\n");
    game_delete(g);
    return false;
  }

  if (!game_is_wrapping(g)) {
    printf("The property wrapping of loaded game are not valid!\n");
    game_delete(g);
    return false;
  }

  if (game_nb_moves_max(g) != 17) {
    printf("The nb_moves_max of loaded game are not valid!\n");
    game_delete(g);
    return false;
  }

  if (game_nb_moves_cur(g)) {
    printf("The nb_moves_cur of loaded game are not valid!\n");
    game_delete(g);
    return false;
  }

  color cells[] = {0, 0, 0, 2, 0, 2, 1, 0, 1, 0, 3, 0, 0, 3, 3, 1, 1, 1, 1, 3};

  for (uint i = 0; i < 20; i++)
    if (cells[i] !=
        game_cell_current_color(g, i % game_width(g), i / game_width(g))) {
      printf("The grid of loaded game are not valid!\n");
      game_delete(g);
      return false;
    }

  game_delete(g);

  // test with bad file : bad is_swap

  // open new file for test
  file = fopen("data/savetest.rec", "w");
  if (file == NULL) {
    printf("The test file couldn't be create!\n");
    return false;
  }

  // fill test file with bad is_swap
  fprintf(file, "5 4 17 V\n0 0 0 2 0\n2 1 0 1 0\n3 0 0 3 3\n1 1 1 1 3\n");
  fclose(file);

  // load file
  g = game_load("data/savetest.rec");
  if (g != NULL) {
    printf("The game has load when can not be!\n");
    remove("data/savetest.rec");
    game_delete(g);
    return false;
  }

  // test with bad file : bad width

  // open new file for test
  file = fopen("data/savetest.rec", "w");
  if (file == NULL) {
    printf("The test file couldn't be create!\n");
    return false;
  }

  // fill test file with bad width
  fprintf(file, "5 4 17 N\n0 0 0 2 0\n2 1 0 1 0 8\n3 0 0 3\n1 1 1 1 3 5\n");
  fclose(file);

  // load file
  g = game_load("data/savetest.rec");
  if (g != NULL) {
    printf("The game has load when can not be!\n");
    remove("data/savetest.rec");
    game_delete(g);
    return false;
  }

  // test with bad file : bad hight

  // open new file for test
  file = fopen("data/savetest.rec", "w");
  if (file == NULL) {
    printf("The test file couldn't be create!\n");
    return false;
  }

  // fill test file with bad height
  fprintf(file,
          "5 4 17 N\n0 0 0 2 0\n2 1 0 1 0\n3 0 0 3 3\n1 1 1 1 3\n1 1 1 1 3\n");
  fclose(file);

  // load file
  g = game_load("data/savetest.rec");
  if (g != NULL) {
    printf("The game has load when can not be!\n");
    remove("data/savetest.rec");
    game_delete(g);
    return false;
  }
  return true;
}

/**
 * @brief Unite test for game_save and game_load
 *
 * @return bool
 */
bool test_game_save_load() {
  // create new game
  color cells[] = {0, 10, 0, 2, 0, 2, 1, 0, 1,  0, 3, 8, 0, 3,
                   3, 1,  5, 1, 1, 3, 0, 0, 11, 2, 0, 2, 1, 0,
                   1, 0,  3, 0, 6, 3, 3, 1, 1,  1, 1, 3};
  game g = game_new_ext(10, 4, cells, 12, false);
  if (g == NULL) {
    fprintf(stderr, "Error: invalid new game!\n");

    return false;
  }

  // save the new game
  game_save(g, "data/savetest.rec");

  // load the save file
  game gl = game_load("data/savetest.rec");
  if (gl == NULL) {
    printf("The game has not been loaded!\n");
    remove("data/savetest.rec");
    return false;
  }

  // remove file
  remove("data/savetest.rec");

  // test if game and loaded game are equal
  if (game_width(g) != game_width(gl)) {
    printf("The width of game and width of loaded game are not equal!\n");
    game_delete(g);
    game_delete(gl);
    return false;
  }

  if (game_height(g) != game_height(gl)) {
    printf("The height of game and height of loaded game are not equal!\n");
    game_delete(g);
    game_delete(gl);
    return false;
  }

  if (game_is_wrapping(g) != game_is_wrapping(gl)) {
    printf(
        "The property wrapping of game and property wrapping of loaded game "
        "are not equal!\n");
    game_delete(g);
    game_delete(gl);
    return false;
  }

  if (game_nb_moves_max(g) != game_nb_moves_max(gl)) {
    printf(
        "The nb_moves_max of game and nb_moves_max of loaded game are not "
        "equal!\n");
    game_delete(g);
    game_delete(gl);
    return false;
  }

  if (game_nb_moves_cur(g) != game_nb_moves_cur(gl)) {
    printf(
        "The nb_moves_cur of game and nb_moves_cur of loaded game are not "
        "equal!\n");
    game_delete(g);
    game_delete(gl);
    return false;
  }

  for (uint i = 0; i < 40; i++)
    if (game_cell_current_color(g, i % game_width(g), i / game_width(g)) !=
        game_cell_current_color(gl, i % game_width(gl), i / game_width(gl))) {
      printf("The grid of game and grid of loaded game are not equal!\n");
      game_delete(g);
      game_delete(gl);
      return false;
    }

  game_delete(g);
  game_delete(gl);

  return true;
}

/**
 * @brief Unite test for game_load and game_save
 *
 * @return bool
 */
bool test_game_load_save() {
  // create file with game inner
  char validfilecontent[] =
      "5 4 7 N\n0 0 0 2 0\n2 1 0 1 0\n3 0 0 3 3\n1 1 1 1 3\n";

  FILE *file;
  file = fopen("data/savetest.rec", "w");
  if (file == NULL) {
    printf("The test file couldn't be create!\n");
    return false;
  }

  fprintf(file, "5 4 7 N\n0 0 0 2 0\n2 1 0 1 0\n3 0 0 3 3\n1 1 1 1 3\n");
  fclose(file);

  // load game from file
  game g = game_load("data/savetest.rec");
  if (g == NULL) {
    printf("The game has not been loaded!\n");
    remove("data/savetest.rec");
    return false;
  }
  remove("data/savetest.rec");

  // save game in new file
  game_save(g, "data/savetest.rec");
  game_delete(g);

  // open generated file
  // file = NULL;
  file = fopen("data/savetest.rec", "r");
  if (file == NULL) {
    printf("The file couldn't be open!\n");
    remove("data/savetest.rec");
    return false;
  }

  char filecontent[49];
  if (!fscanf(file, "%49c", filecontent)) {  // load file content
    fprintf(stderr, "Error: unknown!\n");
    // close and remove file
    fclose(file);
    remove("data/savetest.rec");
    return false;
  }
  filecontent[48] = '\0';

  // close and remove file
  fclose(file);
  remove("data/savetest.rec");

  // compare generate file content with valid default file content
  if (strcmp(filecontent, validfilecontent)) {
    fprintf(stderr, "Error: the game is not correctly load or save!\n");
    return false;
  }

  return true;
}

int main(int argc, char const *argv[]) {
  bool ok = false;

  // in case if program is run without args
  if (argc == 1) {
    /* fprintf(stderr, "Usage: %s <testname> [<...>]\n", argv[0]);
    exit(EXIT_FAILURE);*/

    ok = test_game_copy() && test_game_delete() && test_game_is_over() &&
         test_game_restart() && test_game_restart() && test_game_save() &&
         test_game_load() && test_game_save_load() && test_game_load_save();

  } else {
    // select test from args
    if (!strcmp(argv[1], "copy"))
      ok = test_game_copy();
    else if (!strcmp(argv[1], "delete"))
      ok = test_game_delete();
    else if (!strcmp(argv[1], "is_over"))
      ok = test_game_is_over();
    else if (!strcmp(argv[1], "restart"))
      ok = test_game_restart();
    else if (!strcmp(argv[1], "new_ext"))
      ok = test_game_new_ext();
    else if (!strcmp(argv[1], "save"))
      ok = test_game_save();
    else if (!strcmp(argv[1], "load"))
      ok = test_game_load();
    else if (!strcmp(argv[1], "save2load"))
      ok = test_game_save_load();
    else if (!strcmp(argv[1], "load2save"))
      ok = test_game_load_save();
    else {
      fprintf(stderr, "Error: test \"%s\" not found!\n", argv[1]);
      exit(EXIT_FAILURE);
    }
  }

  // print test result
  if (ok) {
    fprintf(stderr, "Test \"%s\" finished: SUCCESS\n", argv[1]);
    return EXIT_SUCCESS;
  } else {
    fprintf(stderr, "Test \"%s\" finished: FAILURE\n", argv[1]);
    return EXIT_FAILURE;
  }
}
