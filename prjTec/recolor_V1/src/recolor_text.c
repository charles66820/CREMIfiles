/**
 * Author: Arthur BLONDEAU, Charles GOEDEFROIT et Victor ANDRAULT
 */

#include <stdio.h>
#include <stdlib.h>
#include "game.h"

/**
 * @brief Print cells in stdout
 *
 * @param g game with cells to print
 */
void showCells(game g) {
  for (int y = 0; y < SIZE; y++) {
    for (int x = 0; x < SIZE; x++)
      printf("%d", game_cell_current_color(g, x, y));
    printf("\n");
  }
}

/**
 * @brief Print current game state
 *
 * @param g game to print
 */
void printGame(game g) {
  printf("nb coups joués: %d ; nb coups max : %d\n", game_nb_moves_cur(g),
         game_nb_moves_max(g));

  showCells(g);  // print cells

  printf(
      "Jouer un coup: (0,1,2,3,r ou q ; r pour redémarrer ou q pour "
      "quitter)\n");
}

int charToInt(char c) { return c - '0'; }

int main(void) {
  // Init game vars
  bool over = false;
  int nbMaxHit = 12;

  color cells[SIZE * SIZE] = {
      0, 0, 0, 2, 0, 2, 1, 0, 1, 0, 3, 0, 0, 3, 3, 1, 1, 1, 1, 3, 2, 0, 1, 0,
      1, 0, 1, 2, 3, 2, 3, 2, 0, 3, 3, 2, 2, 3, 1, 0, 3, 2, 1, 1, 1, 2, 2, 0,
      2, 1, 2, 3, 3, 3, 3, 2, 0, 1, 0, 0, 0, 3, 3, 0, 1, 1, 2, 3, 3, 2, 1, 3,
      1, 1, 2, 2, 2, 0, 0, 1, 3, 1, 1, 2, 1, 3, 1, 3, 1, 0, 1, 0, 1, 3, 3, 3,
      0, 3, 0, 1, 0, 0, 2, 1, 1, 1, 3, 0, 1, 3, 1, 0, 0, 0, 3, 2, 3, 1, 0, 0,
      1, 3, 3, 1, 1, 2, 2, 3, 2, 0, 0, 2, 2, 0, 2, 3, 0, 1, 1, 1, 2, 3, 0, 1};

  // Create new game
  game g = game_new(cells, nbMaxHit);

  // Show the game for the first time
  printGame(g);
  printf("\n");
  // Game loop
  while (!over) {
    int input = getchar();
    char choice = (char)input;

    // user inputs
    if (choice == 'r') {  // For game restart
      game_restart(g);
      printGame(g);
      printf("\n");
    } else if (choice == 'q') {  // For quit game
      printGame(g);
      printf("DOMMAGE\n");
      game_delete(g);
      exit(EXIT_SUCCESS);
    } else if (charToInt(choice) >= 0 &&
               charToInt(choice) < NB_COLORS) {  // For play shot
      game_play_one_move(g, (color)charToInt(choice));
      printGame(g);
      printf("\n");
    }

    // If the game is lost
    if (game_nb_moves_cur(g) >= game_nb_moves_max(g) && !game_is_over(g)) {
      printf("DOMMAGE\n");
      game_delete(g);
      exit(EXIT_SUCCESS);
    } else  // If the game is over
      over = game_is_over(g);
  }

  // On game is successfully finish
  printf("BRAVO\n");
  showCells(g);

  // Free memory
  game_delete(g);

  return EXIT_SUCCESS;
}
