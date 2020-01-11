#include "game.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

struct game_s {
  color *tab;          // The tab that contains the game cells
  uint nb_moves_max;   // The Maximum amount of move
  uint current_moves;  // The actual amount of move
  color *tab_init;     // The tab_init that contains the copyt of game cells
  uint width;          // The number of columns on the grid
  uint height;         // The number of rows on the grid
  bool wrapping;  // true the game is wrapping, false if the game not wrapping
};

/**
 * @brief Create a new game
 *
 * @param cells tab of colors
 * @param nb_moves_max number of max hit
 * @return game
 * @pre @p cells != NULL
 * @pre @p nb_moves_max > 0
 */
game game_new(color *cells, uint nb_moves_max) {
  return game_new_ext(SIZE, SIZE, cells, nb_moves_max, false);
}

/**
 * @brief create new empty game
 *
 * @return game new empty game
 */
game game_new_empty() {
  return game_new_empty_ext(SIZE, SIZE, false);
}

/**
 * @brief Define cell color in game
 *
 * @param g game for set cell
 * @param x coordinate x
 * @param y coordinate y
 * @param c new color
 * @pre @p g != NULL
 * @pre @p x >= 0
 * @pre @p y >= 0
 * @pre @p c > 0 && c < NB_COLORS
 */
void game_set_cell_init(game g, uint x, uint y, color c) {
  if (g == NULL || c >= NB_COLORS || x >= g->width || y >= g->height) {
    fprintf(stderr, "Bad parameter");
    if (g != NULL) game_delete(g);
    exit(EXIT_FAILURE);
  }

  g->tab[(y * g->width) + x] = c;
  g->tab_init[(y * g->width) + x] = c;
}

/**
 * @brief Set the maximum amount of move
 *
 * @param g The data of the game
 * @param nb_moves_max number of max hit
 * @pre @p g != NULL
 * @pre @p nb_moves_max > 0
 */
void game_set_max_moves(game g, uint nb_max_moves) {
  if (g == NULL || nb_max_moves <= 0) {  // Check in case of bug
    fprintf(stderr,
            " g is NULL or nb_max_moves is set at a number less or equal to 0 "
            ": in the function game_set_max_moves");
    exit(EXIT_FAILURE);
  }
  g->nb_moves_max = nb_max_moves;
}

/**
 * @brief give the max amount of moves
 *
 * @param g The data of the game
 * @return nb_moves_max of g
 * @pre @p g != NULL
 */
uint game_nb_moves_max(cgame g) {
  if (g == NULL) {  // Check in case of bug
    fprintf(stderr, "g is NULL : in the function game_nb_moves_max");
    exit(EXIT_FAILURE);
  }
  return g->nb_moves_max;
}

/**
 * @brief give the current color in coordonate x, y
 *
 * @param g The data of the game
 * @return nb_moves_max of g
 * @pre @p g != NULL
 * @pre @p x > 0
 * @pre @p y > 0
 */
color game_cell_current_color(cgame g, uint x, uint y) {
  if (g == NULL || x >= (g->width) || y >= g->height) {
    exit(EXIT_FAILURE);
  }
  return g->tab[x + y * (g->width)];
}

/**
 * @brief give the number of the actual moves
 *
 * @param g The data of the game
 * @return current_moves of g
 * @pre @p g != NULL
 */
uint game_nb_moves_cur(cgame g) {
  if (g == NULL) {  // Check in case of bug
    fprintf(stderr, "g is NULL : in the function game_nb_moves_cur");
    exit(EXIT_FAILURE);
  }
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
 * @pre @p tc > 0 && tc < NB_COLORS
 * @pre @p c > 0 && c < NB_COLORS
 */
void ff(game g, uint x, uint y, color tc, color c) {
  if (g == NULL || tc >= NB_COLORS || c >= NB_COLORS) {
    fprintf(stderr, "Bad parameter");
    if (g != NULL) game_delete(g);
    exit(EXIT_FAILURE);
  }

  if (x >= g->width || y >= g->height || g->tab[(y * g->width) + x] == c)
    return;
  if (g->tab[(y * g->width) + x] != tc) return;

  g->tab[(y * g->width) + x] = c;  // replace target color by color

  if (g->wrapping) {
    ff(g, (x + 1) % g->width, y, tc, c);   // spread to right
    ff(g, x, (y + 1) % g->height, tc, c);  // spread to down
    if (x != 0)                            // spread to left
      ff(g, x - 1, y, tc, c);
    else
      ff(g, g->width, y, tc, c);
    if (y != 0)                            // spread to up
      ff(g, x, y - 1, tc, c);
    else
      ff(g, x, g->height, tc, c);
  } else {
    ff(g, x + 1, y, tc, c);              // spread to right
    ff(g, x, y + 1, tc, c);              // spread to down
    if (x != 0) ff(g, x - 1, y, tc, c);  // spread to left
    if (y != 0) ff(g, x, y - 1, tc, c);  // spread to up
  }
}

/**
 * @brief For play one move in game and propag color
 *
 * @param g game to play one move
 * @param c color played
 * @pre @p g != NULL
 * @pre @p c > 0 && c < NB_COLORS
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

/**
 * @brief copy game in parameter
 *
 * @param g game for copy
 * @return game new game copy from game in parameter
 * @pre @p g != NULL
 */
game game_copy(cgame g) {
  if (g == NULL || g->tab == NULL || g->tab_init == NULL) {
    exit(EXIT_FAILURE);
  }
  game game_copy = (game)malloc(sizeof(struct game_s));
  if (game_copy == NULL) {
    exit(EXIT_FAILURE);
  }
  game_copy->tab = malloc(g->width * g->height * sizeof(color));
  game_copy->tab_init = malloc(g->width * g->height * sizeof(color));
  if (game_copy->tab == NULL || game_copy->tab_init == NULL) {
    game_delete(game_copy);
    exit(EXIT_FAILURE);
  }
  for (int i = 0; i < g->width * g->height; i++) {
    game_copy->tab[i] = g->tab[i];
    game_copy->tab_init[i] = g->tab_init[i];
  }
  game_copy->nb_moves_max = g->nb_moves_max;
  game_copy->current_moves = g->current_moves;
  game_copy->width = g->width;
  game_copy->height = g->height;
  game_copy->wrapping = g->wrapping;
  return game_copy;
}

/**
 * @brief delete, by using free(), the game from the memory
 *
 * @param g The all the data of the game
 * @pre @p stderr, "g is NULL : in the function game_delete"
 **/
void game_delete(game g) {
  if (g == NULL) exit(EXIT_FAILURE);
  free(g->tab);
  free(g->tab_init);
  free(g);
}

/**
 * @brief test if the game is over or not
 *
 * @param g The data of the game
 * @return a boolean, false if game is over, true otherwise
 * @pre @p g != NULL
 */
bool game_is_over(cgame g) {
  if (g == NULL) {
    exit(EXIT_FAILURE);
  }
  color ref = g->tab[0];
  bool over = true;
  if (g->current_moves > g->nb_moves_max) {
    over = false;
  }
  for (int i = 0; i < (g->width) * (g->height); i++) {
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
 * @pre @p g != NULL
 */
void game_restart(game g) {
  if (g == NULL) {
    fprintf(stderr, "Bad parameter");
    exit(EXIT_FAILURE);
  }

  g->current_moves = 0;

  // Copy initial color cells to game cells
  for (uint i = 0; i < g->width * g->height; i++) {
    g->tab[i] = g->tab_init[i];
  }
}

/**
 * @brief Creates a new empty game having height rows and width
 * columns. All the cells will have the default color (whose value is
 * 0). The maximum number of moves is set to 0
 * @param width the width of the grid
 * @param height the height of the grid
 * @param wrapping whether or not the game is wrapping
 * @return the created game
 * @pre @p width > 0
 * @pre @p height > 0
 * @pre @p wrapping == false || true
 **/
game game_new_empty_ext(uint width, uint height, bool wrapping) {
  if (width < 1 || height < 1) {
    fprintf(stderr, "Invalid parameter on the function 'game_new_empty_ext");
    exit(EXIT_FAILURE);
  }

  game new_game = (game)malloc(sizeof(struct game_s));
  new_game->tab = malloc(new_game->width * new_game->height * sizeof(color));
  new_game->tab_init =
      malloc(new_game->width * new_game->height * sizeof(color));
  if (new_game == NULL || new_game->tab == NULL || new_game->tab_init == NULL) {
    exit(EXIT_FAILURE);
  }
  new_game->width = width;
  new_game->height = height;
  new_game->wrapping = wrapping;
  return new_game;
}

/**
 * @brief Returns the number of columns on the game
 * @param game the game
 * @return the width of the game
 * @pre @p g is a valid pointer toward a cgame structure
 **/
uint game_width(cgame game) {
  if (game == NULL) {
    exit(EXIT_FAILURE);
  }
  return game->width;
}

/**
 * @brief Creates a new game
 * @param width the width of the grid
 * @param height the height of the grid
 * @param cells 1D array describing the color of each cell of the game. The
 *storage is row by row
 * @param wrapping whether or not the game is wrapping
 * @return the created game
 * @pre @p width > 0
 * @pre @p height > 0
 * @pre @p cells != NULL
 * @pre @p nb_moves_max > 0
 * @pre @p wrapping == false || true
 **/
game game_new_ext(uint width, uint height, color *cells, uint nb_moves_max,
                  bool wrapping) {
  if (cells == NULL || nb_moves_max <= 0 || width < 1 || height < 1) {
    fprintf(stderr, "Invalid parameter");
    exit(EXIT_FAILURE);
  }

  game g = malloc(sizeof(struct game_s));
  if (g == NULL) {
    fprintf(stderr, "Not enough memory");
    exit(EXIT_FAILURE);
  }

  g->nb_moves_max = nb_moves_max;
  g->current_moves = 0;
  g->width = width;
  g->height = height;
  g->wrapping = wrapping;

  g->tab = malloc((g->width * g->height) * sizeof(color));
  if (g->tab == NULL) {
    fprintf(stderr, "Not enough memory");
    game_delete(g);
    exit(EXIT_FAILURE);
  }

  g->tab_init = malloc((g->width * g->height) * sizeof(color));
  if (g->tab_init == NULL) {
    fprintf(stderr, "Not enough memory");
    game_delete(g);
    exit(EXIT_FAILURE);
  }

  for (uint i = 0; i < g->width * g->height; i++) {
    g->tab[i] = cells[i];
    g->tab_init[i] = cells[i];
  }

  return g;
}
