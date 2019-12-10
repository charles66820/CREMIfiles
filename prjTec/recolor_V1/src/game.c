#include "game.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

struct game_s {
  color *tab;          // le tableau contenant les cases du jeux
  uint nb_moves_max;   // nombre de coups max
  uint current_moves;  // nombre de coups actuels
  color *tab_init;
  uint size;
};

/**
 * @brief Create a new game
 *
 * @param cells tab of colors
 * @param nb_moves_max number of max hit
 * @return game
 */
game game_new(color *cells, uint nb_moves_max) {
  if (cells == NULL) {
    fprintf(stderr, "Bad parameter");
    exit(EXIT_FAILURE);
  }

  game g = malloc(sizeof(struct game_s));
  if (g == NULL) {
    fprintf(stderr, "Not enough memory");
    exit(EXIT_FAILURE);
  }

  g->nb_moves_max = nb_moves_max;
  g->current_moves = 0;
  g->size = SIZE;

  g->tab = malloc((SIZE * SIZE) * sizeof(color));
  if (g->tab == NULL) {
    fprintf(stderr, "Not enough memory");
    game_delete(g);
    exit(EXIT_FAILURE);
  }

  g->tab_init = malloc((SIZE * SIZE) * sizeof(color));
  if (g->tab_init == NULL) {
    fprintf(stderr, "Not enough memory");
    game_delete(g);
    exit(EXIT_FAILURE);
  }

  for (uint i = 0; i < SIZE * SIZE; i++) {
    g->tab[i] = cells[i];
    g->tab_init[i] = cells[i];
  }

  return g;
}

game game_new_empty() {
  color *tab = (color *)malloc(SIZE * SIZE * sizeof(color));
  if (tab == NULL){
    exit(EXIT_FAILURE);
  }
  for (int i = 0; i < SIZE * SIZE; i++) {
    tab[i] = 0;
  }
  game game_empty = (game)malloc(sizeof(game));
  if (game_empty == NULL){
    exit(EXIT_FAILURE);
  }
  game_empty->tab = tab;
  game_empty->nb_moves_max = 0;
  game_empty->current_moves = 0;
  game_empty->tab_init = tab;
  game_empty->size = SIZE;
  return game_empty;
}

/**
 * @brief Define cell color in game
 *
 * @param g game for set cell
 * @param x coordinate x
 * @param y coordinate y
 * @param c new color
 */
void game_set_cell_init(game g, uint x, uint y, color c) {
  if (g == NULL || c >= NB_COLORS || x >= g->size || y >= g->size) {
    fprintf(stderr, "Bad parameter");
    if (g != NULL) game_delete(g);
    exit(EXIT_FAILURE);
  }

  g->tab[(y * g->size) + x] = c;
  g->tab_init[(y * g->size) + x] = c;
}

void game_set_max_moves(game g, uint nb_max_moves) {
  if (g == NULL || nb_max_moves <= 0) exit(EXIT_FAILURE);
  g->nb_moves_max = nb_max_moves;
  return;
}

uint game_nb_moves_max(cgame g) {
  if (g == NULL) exit(EXIT_FAILURE);
  return g->nb_moves_max;
}

color game_cell_current_color(cgame g, uint x, uint y) {
  if (g == NULL || x >= (g->size) || y >= g->size) {
    exit(EXIT_FAILURE);
  }
  return g->tab[x + y * (g->size)];
}

uint game_nb_moves_cur(cgame g) {
  if (g == NULL) exit(EXIT_FAILURE);
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
 */
void ff(game g, uint x, uint y, color tc, color c) {
  if (g == NULL || tc >= NB_COLORS || c >= NB_COLORS) {
    fprintf(stderr, "Bad parameter");
    if (g != NULL) game_delete(g);
    exit(EXIT_FAILURE);
  }

  if (x >= g->size || y >= g->size || g->tab[(y * g->size) + x] == c) return;
  if (g->tab[(y * g->size) + x] != tc) return;

  g->tab[(y * g->size) + x] = c;  // replace target color by color

  ff(g, x + 1, y, tc, c);                            // spread to right
  ff(g, x + 1, y + 1, tc, c);                        // spread to right-down
  ff(g, x, y + 1, tc, c);                            // spread to down
  if (x != 0) ff(g, x - 1, y + 1, tc, c);            // spread to left-down
  if (x != 0) ff(g, x - 1, y, tc, c);                // spread to left
  if (x != 0 && y != 0) ff(g, x - 1, y - 1, tc, c);  // spread to left-up
  if (y != 0) ff(g, x, y - 1, tc, c);                // spread to up
  if (y != 0) ff(g, x + 1, y - 1, tc, c);            // spread to up-right
}

/**
 * @brief For play one move in game and propag color
 *
 * @param g game to play one move
 * @param c color played
 */
void game_play_one_move(game g, color c) {
  if (g == NULL || c >= NB_COLORS) {
    fprintf(stderr, "Bad parameter");
    if (g != NULL) game_delete(g);
    exit(EXIT_FAILURE);
  }

  // Test if play can play
  if (g->current_moves < g->nb_moves_max) {
    ff(g, 0, 0, g->tab[0], c);
    g->current_moves++;
  }
}

game game_copy(cgame g) {
  if (g == NULL || g->tab == NULL || g->tab_init == NULL) {
    exit(EXIT_FAILURE);
  }
  game game_copy = (game)malloc(sizeof(struct game_s));
  if (game_copy == NULL) {
    exit(EXIT_FAILURE);
  }
  game_copy->tab = malloc(SIZE * SIZE * sizeof(color));
  game_copy->tab_init = malloc(SIZE * SIZE * sizeof(color));
  if (game_copy->tab == NULL || game_copy->tab_init == NULL) {
    game_delete(game_copy);
    exit(EXIT_FAILURE);
  }
  for (int i = 0; i < (g->size) * (g->size); i++) {
    game_copy->tab[i] = g->tab[i];
    game_copy->tab_init[i] = g->tab_init[i];
  }
  game_copy->nb_moves_max = g->nb_moves_max;
  game_copy->current_moves = g->current_moves;
  game_copy->size = g->size;
  return game_copy;
}

void game_delete(game g) {
  if (g == NULL) exit(EXIT_FAILURE);
  free(g->tab);
  free(g->tab_init);
  free(g);
}

bool game_is_over(cgame g) {
  if (g == NULL) {
    exit(EXIT_FAILURE);
  }
  color ref = g->tab[0];
  bool over = true;
  if (g->current_moves > g->nb_moves_max) {
    over = false;
  }
  for (int i = 0; i < (g->size) * (g->size); i++) {
    if (g->tab[i] != ref) {
      over = false;
    }
  }
  return over;
}

/**
 * @brief Reinit game
 *
 * @param g game for restart
 */
void game_restart(game g) {
  if (g == NULL) {
    fprintf(stderr, "Bad parameter");
    exit(EXIT_FAILURE);
  }

  g->current_moves = 0;

  // Copy initial color cells to game cells
  for (uint i = 0; i < g->size * g->size; i++) {
    g->tab[i] = g->tab_init[i];
  }
}