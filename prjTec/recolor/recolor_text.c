/**
 * Author: Arthur BLONDEAU, Charles GOEDEFROIT et Victor ANDRAULT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "game.h"
#include "game_io.h"

/**
 * @brief Print cells in stdout
 *
 * @param g game with cells to print
 */
void showCells(game g) {
  color color;
  char sColor[10] = "";
  for (int y = 0; y < game_height(g); y++) {
    for (int x = 0; x < game_width(g); x++) {
      color = game_cell_current_color(g, x, y);
      if (color <= 9) {
        sprintf(sColor, "%u", color);
      } else {
        sprintf(sColor, "%c", 55 + color);
      }
      printf("%s", sColor);
    }
    printf("\n");
  }
}

/**
 * @brief Print current game state
 *
 * @param g game to print
 */
void printGame(game g) {
  printf("nb coups joués: %u ; nb coups max : %u\n", game_nb_moves_cur(g),
         game_nb_moves_max(g));

  showCells(g);  // print cells

  printf(
      "Jouer un coup: (0,1,2,3,r ou q ; r pour redémarrer, q pour quitter ou s "
      "pour enregistré)\n");
}

int charToInt(char c) { return c - '0'; }

int main(int argc, char* argv[]) {
  // Init game vars
  bool pl = true;
  game g = NULL;

  if (argc > 1) {
    g = game_load(argv[1]);
    if (g == NULL) printf("Error on game load : The default game as load\n");
  }

  if (argc == 1 || g == NULL) {  // if game is launch without arguments or if
                                 // game is null we create new game
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

  // Show the game for the first time
  printGame(g);
  printf("\n");
  // Game loop
  while (true) {
    int input = getchar();
    char choice = (char)input;

    // user inputs
    if (choice == 'r') {  // For game restart
      pl = true;
      game_restart(g);
      printGame(g);
      printf("\n");

    } else if (choice == 'q') {  // For quit game
      printGame(g);
      printf("DOMMAGE\n");
      game_delete(g);
      exit(EXIT_SUCCESS);

    } else if (choice == 's') {  // For save game
      char fileName[251];
      printf("Saisiser le nom du fichier où sera enregistré le jeu : ");
      scanf("%250s", fileName);
      strcat(fileName, ".rec");
      game_save(g, fileName);
      printf("Partie enregistré dans le fichier %s!\n", fileName);
    } else if (charToInt(choice) >= 0 &&
               charToInt(choice) <= 9) {  // For play shot
      game_play_one_move(g, (color)charToInt(choice));
      printGame(g);
      printf("\n");
    } else if (choice >= 65 && choice < 71) {
      game_play_one_move(g, (color)choice - 55);
      printGame(g);
      printf("\n");
    }

    // If the game is lost
    if (game_nb_moves_cur(g) >= game_nb_moves_max(g) && !game_is_over(g) &&
        pl) {
      printf("DOMMAGE\n");
      pl = false;
    }

    if (game_is_over(g)) {
      // On game is successfully finish
      printf("BRAVO\n");
      showCells(g);

      // exit on win
      game_delete(g);
      exit(EXIT_SUCCESS);
    }
  }

  // Free memory
  game_delete(g);

  return EXIT_SUCCESS;
}
