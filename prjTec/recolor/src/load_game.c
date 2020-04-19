#include "load_game.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef SDL
#include "SDL_model.h"
#endif
#include "game.h"
#include "game_io.h"
#include "game_rand.h"

game load_game(int argc, char* argv[]) {
  game g = NULL;
  if (argc == 2) {
    g = game_load(argv[1]);
    if (!g)
#ifdef SDL
      PRINT("Game error", "Error on game load : The default game as load\n");
#else
      fprintf(stderr, "Error on game load : The default game as load\n");
#endif
  }

  if (argc == 3 || argc > 6)
#ifdef SDL
    PRINT("Game error", "Invalid parameters. Loading default game...\n");
#else
    fprintf(stderr, "Invalid parameters. Loading default game...\n");
#endif

  if (argc >= 4 && argc <= 6) {  // create_rand_game
    int width = atoi(argv[1]);
    int height = atoi(argv[2]);
    int nb_moves_max = atoi(argv[3]);
    int nb_colors = 4;
    bool wrapping = false;
    if (width <= 0 || height <= 0 || nb_moves_max <= 0)
      fprintf(stderr, "Invalid parameters. Loading default game...\n");
    else if (argc == 4)
      g = game_random_ext(width, height, nb_moves_max, nb_colors, wrapping);
    else if (argc == 5) {
      if (argv[4][0] == 'N')
        wrapping = false;
      else if (argv[4][0] == 'S')
        wrapping = true;
      else
        nb_colors = atoi(argv[4]);
      if (nb_colors >= 2 && nb_colors < 17)
        g = game_random_ext(width, height, nb_moves_max, nb_colors, wrapping);
      else
        fprintf(stderr, "Invalid parameters. Loading default game...\n");
    } else if (argc == 6) {
      nb_colors = atoi(argv[4]);
      if (nb_colors >= 2 && nb_colors < 17 &&
          (argv[5][0] == 'N' || argv[5][0] == 'S'))
        g = game_random_ext(width, height, nb_moves_max, nb_colors,
                            argv[5][0] == 'S');
      else
        fprintf(stderr, "Invalid parameters. Loading default game...\n");
    }
  }

  if (argc == 1 || !g) {  // if game is launch without arguments or
                          // if game is null we create new game
    int nbMaxHit = 12;

    color cells[SIZE * SIZE] = {
        0, 0, 0, 2, 0, 2, 1, 0, 1, 0, 3, 0, 0, 3, 3, 1, 1, 1, 1, 3, 2, 0, 1, 0,
        1, 0, 1, 2, 3, 2, 3, 2, 0, 3, 3, 2, 2, 3, 1, 0, 3, 2, 1, 1, 1, 2, 2, 0,
        2, 1, 2, 3, 3, 3, 3, 2, 0, 1, 0, 0, 0, 3, 3, 0, 1, 1, 2, 3, 3, 2, 1, 3,
        1, 1, 2, 2, 2, 0, 0, 1, 3, 1, 1, 2, 1, 3, 1, 3, 1, 0, 1, 0, 1, 3, 3, 3,
        0, 3, 0, 1, 0, 0, 2, 1, 1, 1, 3, 0, 1, 3, 1, 0, 0, 0, 3, 2, 3, 1, 0, 0,
        1, 3, 3, 1, 1, 2, 2, 3, 2, 0, 0, 2, 2, 0, 2, 3, 0, 1, 1, 1, 2, 3, 0, 1};

    // Create new game
    g = game_new(cells, nbMaxHit);
  }
  return g;
}