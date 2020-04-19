#include "game_rand.h"
#include <stdbool.h>
#include <stdlib.h>
#include "game.h"

game game_random_ext(uint width, uint height, uint nb_max_moves, uint max_color,
                     bool is_wrapping) {
  game g = game_new_empty_ext(width, height, is_wrapping);
  for (uint x = 0; x < width; x++)
    for (uint y = 0; y < height; y++) {
      uint c = (uint)rand() % max_color;
      game_set_cell_init(g, x, y, c);
    }
  game_restart(g);
  game_set_max_moves(g, nb_max_moves);
  return g;
}
