#ifndef GAME_RAND_H
#define GAME_RAND_H

#include "game.h"

/**
 * @brief Computes a random game. This function insures that every game of a
 *given size can be generate, but not necessarily with the same probability.
 * @param width width of the game
 * @param height height of the game
 * @param is_wrapping
 * @param max_color number of color used
 * @param nb_max_moves
 * @return a random game.
 * @post the result is a game. This game doesn't necessarily admit a solution.
 **/
game game_random_ext(uint width, uint height, uint nb_max_moves, uint max_color,
                     bool is_wrapping);

#endif  // GAME_RAND
