#ifndef __GAME_H__
#define __GAME_H__
#include <stdbool.h>

/**
 * @file game.h
 * @brief This file describes the programming interface of a puzzle
 * game, named 'recolor'.
 *
 * @mainpage
 *
 *
 *  This game starts with the initial grid of multi colored cells. The
 *  cells of the grid are denoted by (x,y) coordinates, (0,0)
 *  corresponding to the top left corner, and the first coordinate
 *  indicates the column.
 *
 * The goal of the game is to color the whole grid with the same color
 * in less than the number of allowed steps. Initially, the cells of the grid have a
 * random color. After each move, the player can fill the top left
 * cell in a color of his choice. Then any cell currently forming a
 * contiguous region with the top left cell (i.e.every cell
 * reachable from the starting cell by a connected path of cells all
 * having the same color) will be repainted with that color and the
 * amount of moves left will be decreased by 1.
 *
 * For further details on the rules, see <a
 * href="https://www.chiark.greenend.org.uk/~sgtatham/puzzles/doc/flood.html">here</a>.
 *
 * *
 * Here is an example of a game and its solution.
 *
 @verbatim

 | Example of game

000202101030
033111132010
101232320332
231032111220
212333320100
033011233213
112220013112
131310101333
030100211130
131000323100
133112232002
202301112301

Nb moves max : 12

Solution: 3 1 3 1 0 3 1 0 1 3 2 0
 @endverbatim

 */

#define SIZE 12

typedef unsigned int uint;

/**
 * @brief Different colors (red=0, green=1, blue=2 or yellow=3) used in the game
 **/
typedef enum color_e {RED, GREEN, BLUE, YELLOW, NB_COLORS} color;

/**
 * @brief The structure pointer that stores the game. To create a game, you can proceed in two ways:
 * just call game_new(),
 * or first create an empty game with game_new_empty(), and then initialize it with game_set_cell_init() and game_set_max_moves().
 **/
typedef struct game_s *game;

/**
 * @brief The structure constant pointer that stores the game
 * That means that it is not possible to modify the game using this pointer.
 * See also: http://www.geeksforgeeks.org/const-qualifier-in-c/
 * See also this more technical discussion:
 *http://stackoverflow.com/questions/8504411/typedef-pointer-const-weirdness
 **/
typedef const struct game_s *cgame;

/**
 * @brief Creates a new game and initializes it in one call.
 * @param cells 1D array describing the color of each cell of the game. The storage is row by row
 * @param nb_max_moves the value of the maximum number of moves
 * @return the created game
 * @pre @p cells is an initialized array of size SIZE*SIZE.
 * @pre @p nb_max_moves > 0
 **/
game game_new(color *cells, uint nb_moves_max);


/**
 * @brief Creates an empty game.
 * @details Creates an empty game having SIZE rows and SIZE columns. All the cells will have the default color
 * (whose value is RED). The maximum number of moves is set to 0.
 * @return The created game
 **/
game game_new_empty();

/**
 * @brief Sets the initial color (and the current color) of the cell located at given coordinates.
 * @param g the game
 * @param x the first coordinate of the cell
 * @param y the second coordinate of the cell
 * @param c color to be used
 * @pre @p g is a valid pointer toward a game structure
 * @pre @p x < SIZE
 * @pre @p y < SIZE
 * @pre 0 <= @p c < NB_COLORS
 **/
void game_set_cell_init(game g, uint x, uint y, color c);

/**
 * @brief Sets the maximum number of moves for the game g
 * @param g the game
 * @param nb_max_moves the value of the maximum number of moves
 * @pre @p g is a valid pointer toward a game structure
 * @pre @p nb_max_moves > 0
 **/
void game_set_max_moves(game g, uint nb_max_moves);

/**
 * @brief Gets the maximum number of moves for the given game.
 * @param g the game
 * @return the value of the maximum number of moves
 * @pre @p g is a valid pointer toward a cgame structure
 **/
uint game_nb_moves_max(cgame g);

/**
 * @brief Gets the color of the cell located at given coordinates.
 * @param g the game
 * @param x the first coordinate of the cell
 * @param y the second coordinate of the cell
 * @return the color of the cell
 * @pre @p g is a valid pointer toward a cgame structure
 * @pre @p x < SIZE
 * @pre @p y < SIZE
 **/
color game_cell_current_color(cgame g, uint x, uint y);

/**
 * @brief Gets the number of moves since the last start (or restart).
 * @param g the game
 * @return the value of the current number of moves
 * @pre @p g is a valid pointer toward a cgame structure
 **/
uint game_nb_moves_cur(cgame g);

/**
 * @brief Performs one move using color c
 * @param g the game
 * @param c the color
 * @pre @p g is a valid pointer toward a game structure
 * @pre 0 <= @p c < NB_COLORS
 **/
void game_play_one_move(game g, color c);

/**
 * @brief Clones the game g
 * @param g a constant pointer on the game to clone
 * @return the clone of g
 * @pre @p g is a valid pointer toward a cgame structure
 **/
game game_copy(cgame g);

/**
 * @brief Destroys the game and frees the allocated memory
 * @param g the game to destroy
 * @pre @p g is a valid pointer toward a game structure
 **/
void game_delete(game g);

/**
 * @brief Checks if the game is over.
 * @details The game is considered to be over if the all the cells in the grid
 * have the same color and the number of moves is smaller than or
 * equal to the maximum allowed number of moves
 * @return true, if the game is over, false otherwise
 * @param g the game
 * @pre @p g is a valid pointer toward a game structure
 **/
bool game_is_over(cgame g);

/**
 * @brief Restarts a game by resetting the colors of the cells to their
 * initial value and by setting the current number of moves to 0.
 * @param g the game to restart
 * @pre @p g is a valid pointer toward a game structure
 **/
void game_restart(game g);

#endif  // __GAME_H__
