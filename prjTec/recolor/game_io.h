#ifndef GAME_IO_H
#define GAME_IO_H
#include <stdio.h>
#include "game.h"

/**
 * @file game_io.h
 *
 * @brief This file provides functions to load or save a game.
 *
 **/

/**
 * @brief Creates a game by loading its description in a file
 *  NB : THE FORMAT WILL BE SPECIFIED LATER
 * @param filename
 * @return the loaded game
 **/
game game_load(char *filename);

/**
 * @brief Save a game in a file
 *  NB : THE FORMAT WILL BE SPECIFIED LATER
 * @param g game to save
 * @param filename output file
 **/
void game_save(cgame g, char *filename);

#endif  // GAME_IO_H
