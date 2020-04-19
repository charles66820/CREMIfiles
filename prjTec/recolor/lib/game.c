#include "game.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

struct game_s {
  color
      *tab;  // The tab that contains the game cells in a single dimensional tab
  uint nb_moves_max;   // The Maximum amount of move
  uint current_moves;  // The actual amount of move
  color *tab_init;     // The tab_init that contains the copy of game cells in a
                       // single dimensional tab
  uint width;          // The number of columns on the grid
  uint height;         // The number of rows on the grid
  bool wrapping;  // true the game is wrapping, false if the game not wrapping
};

/**
 * @brief Check if pointer is null
 *
 * @param p pointer will be check
 * @param msg message will print if pointer is null
 */
static void check_pointer(const void *p, char *msg) {
  if (p == NULL) {
    if (msg == NULL)
      fprintf(stderr, "Null pointer error.\n");
    else
      fprintf(stderr, "%s\n", msg);
    exit(EXIT_FAILURE);
  }
}

game game_new(color *cells, uint nb_moves_max) {
  return game_new_ext(SIZE, SIZE, cells, nb_moves_max, false);
}

game game_new_empty() { return game_new_empty_ext(SIZE, SIZE, false); }

void game_set_cell_init(game g, uint x, uint y, color c) {
  check_pointer(g, "g parameter on the function game_set_cell_init is null.");

  if (x >= g->width || y >= g->height) {
    fprintf(stderr, "Bad parameter on the function game_set_cell_init.\n");
    game_delete(g);
  }

  g->tab[(y * g->width) + x] = c;
  g->tab_init[(y * g->width) + x] = c;
}

void game_set_max_moves(game g, uint nb_max_moves) {
  check_pointer(g, "g parameter on the function game_set_max_moves is null.\n");
  g->nb_moves_max = nb_max_moves;
}

uint game_nb_moves_max(cgame g) {
  check_pointer(g, "g parameter on the function game_nb_moves_max is null.\n");
  return g->nb_moves_max;
}

color game_cell_current_color(cgame g, uint x, uint y) {
  check_pointer(
      g, "g parameter on the function game_cell_current_color is null.\n");
  if (x >= g->width || y >= g->height) {
    exit(EXIT_FAILURE);
  }
  return (color)g->tab[x + y * g->width];
}

uint game_nb_moves_cur(cgame g) {
  check_pointer(g, "g parameter on the function game_nb_moves_cur is null.\n");
  return g->current_moves;
}

/**
 * @brief Spread color by flood fill algo
 *
 * @param g game object
 * @param x abscissa
 * @param y ordinate
 * @param tc target color
 * @param c color
 * @pre @p g != NULL
 * @pre @p x >= 0
 * @pre @p y >= 0
 * @pre @p tc > 0
 * @pre @p c > 0
 */
static void ff(game g, uint x, uint y, color tc, color c) {
  check_pointer(g, "g parameter on the function ff is null.\n");

  if (x >= g->width || y >= g->height || g->tab[y * g->width + x] == c) return;
  if (g->tab[y * g->width + x] != tc) return;

  g->tab[y * g->width + x] = c;  // replace target color by color

  if (g->wrapping) {
    ff(g, (x + 1) % g->width, y, tc, c);   // spread to right
    ff(g, x, (y + 1) % g->height, tc, c);  // spread to down
    if (x != 0)
      ff(g, x - 1, y, tc, c);  // spread to left
    else
      ff(g, g->width - 1, y, tc, c);
    if (y != 0)  // spread to up
      ff(g, x, y - 1, tc, c);
    else
      ff(g, x, g->height - 1, tc, c);
  } else {
    ff(g, x + 1, y, tc, c);              // spread to right
    ff(g, x, y + 1, tc, c);              // spread to down
    if (x != 0) ff(g, x - 1, y, tc, c);  // spread to left
    if (y != 0) ff(g, x, y - 1, tc, c);  // spread to up
  }
}

void game_play_one_move(game g, color c) {
  check_pointer(g, "g parameter on the function game_play_one_move is null.\n");

  ff(g, 0, 0, (color)g->tab[0], c);

  g->current_moves++;
}

game game_copy(cgame g) {
  check_pointer(g, "g parameter on the function game_copy is null.\n");

  game game_copy = game_new_empty_ext(g->width, g->height, g->wrapping);
  check_pointer(game_copy,
                "game_copy can not be create on the function game_copy.\n");

  for (int i = 0; i < g->width * g->height; i++) {
    game_copy->tab[i] = g->tab[i];
    game_copy->tab_init[i] = g->tab_init[i];
  }
  game_copy->nb_moves_max = g->nb_moves_max;
  game_copy->current_moves = g->current_moves;
  return game_copy;
}

void game_delete(game g) {
  check_pointer(g, "g parameter on the function game_delete is null.\n");

  if (g->tab != NULL) free(g->tab);
  if (g->tab_init != NULL) free(g->tab_init);
  free(g);
}

bool game_is_over(cgame g) {
  check_pointer(g, "g parameter on the function game_is_over is null.\n");

  color ref = g->tab[0];

  for (int i = 0; i < g->width * g->height; i++)
    if (g->tab[i] != ref) return false;

  if (g->current_moves <= g->nb_moves_max) return true;

  return false;
}

void game_restart(game g) {
  check_pointer(g, "g parameter on the function game_restart is null.\n");

  g->current_moves = 0;

  // Copy initial color cells to game cells
  for (uint i = 0; i < g->width * g->height; i++) {
    g->tab[i] = g->tab_init[i];
  }
}

game game_new_empty_ext(uint width, uint height, bool wrapping) {
  color *cells = malloc(width * height * sizeof(color));
  check_pointer(cells,
                "Not enough memory for enough cells on the function "
                "game_new_empty_ext.\n");

  for (uint i = 0; i < width * height; i++) cells[i] = 0;

  game g = game_new_ext(width, height, cells, 0, wrapping);

  free(cells);
  return g;
}

uint game_width(cgame game) {
  check_pointer(game, "g parameter on the function game_width is null.\n");

  return game->width;
}

uint game_height(cgame game) {
  check_pointer(game, "g parameter on the function game_height is null.\n");

  return game->height;
}

bool game_is_wrapping(cgame game) {
  check_pointer(game,
                "g parameter on the function game_is_wrapping is null.\n");

  return game->wrapping;
}

game game_new_ext(uint width, uint height, color *cells, uint nb_moves_max,
                  bool wrapping) {
  check_pointer(cells,
                "cells parameter on the function game_new_ext is null.\n");
  if (width < 1 || height < 1) {
    fprintf(stderr, "Invalid parameter on the function game_new_ext.\n");
    exit(EXIT_FAILURE);
  }

  game g = malloc(sizeof(struct game_s));
  check_pointer(g,
                "Not enough memory for game g allocation on the function "
                "game_new_ext is null.\n");

  g->nb_moves_max = nb_moves_max;
  g->current_moves = 0;
  g->width = width;
  g->height = height;
  g->wrapping = wrapping;

  g->tab = malloc((g->width * g->height) * sizeof(color));
  if (g->tab == NULL) {
    fprintf(stderr,
            "Not enough memory for g->tab in the function game_new_ext.\n");
    game_delete(g);
    exit(EXIT_FAILURE);
  }

  g->tab_init = malloc((g->width * g->height) * sizeof(color));
  if (g->tab_init == NULL) {
    fprintf(
        stderr,
        "Not enough memory for g->tab_init in the function game_new_ext.\n");
    game_delete(g);
    exit(EXIT_FAILURE);
  }

  for (uint i = 0; i < g->width * g->height; i++) {
    g->tab[i] = cells[i];
    g->tab_init[i] = cells[i];
  }

  return g;
}
