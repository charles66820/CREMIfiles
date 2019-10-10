/**
 * Author: Arthur BLONDEAU, Charles GOEDEFROIT, Victor ANDRAULT
 */

#include <stdio.h>
#include <stdlib.h>
#include "game.h"

void showCells(game g)
{
    for (int y = 0; y < SIZE; y++)
    {
        for (int x = 0; x < SIZE; x++)
            printf("%d", game_cell_current_color(g, x, y));
        printf("\n");
    }
}

void printGame(game g) {
    printf("nb coups joués: %d ; nb coups max : %d\n", game_nb_moves_cur(g), game_nb_moves_max(g));

    showCells(g); // affiche la grille

    printf("Jouer un coup: (0,1,2,3,r ou q ; r pour redémarrer ou q pour quitter)\n");
}

int charToInt(char c) {
    return c-'0';
}

int main(void) {
    bool over = false;
    int nbMaxHit = 12;

    color cells[SIZE*SIZE] = {
        0, 0, 0, 2, 0, 2, 1, 0, 1, 0, 3, 0,
        0, 3, 3, 1, 1, 1, 1, 3, 2, 0, 1, 0,
        1, 0, 1, 2, 3, 2, 3, 2, 0, 3, 3, 2,
        2, 3, 1, 0, 3, 2, 1, 1, 1, 2, 2, 0,
        2, 1, 2, 3, 3, 3, 3, 2, 0, 1, 0, 0,
        0, 3, 3, 0, 1, 1, 2, 3, 3, 2, 1, 3,
        1, 1, 2, 2, 2, 0, 0, 1, 3, 1, 1, 2,
        1, 3, 1, 3, 1, 0, 1, 0, 1, 3, 3, 3,
        0, 3, 0, 1, 0, 0, 2, 1, 1, 1, 3, 0,
        1, 3, 1, 0, 0, 0, 3, 2, 3, 1, 0, 0,
        1, 3, 3, 1, 1, 2, 2, 3, 2, 0, 0, 2,
        2, 0, 2, 3, 0, 1, 1, 1, 2, 3, 0, 1};

    game g = game_new(cells, nbMaxHit);

    printGame(g);

    while (!over)
    {
        int input = getchar();
        char choice = (char)input;

        // client inputs
        if (choice == 'r') {
            game_restart(g);
            printGame(g);
        }
        else if (choice == 'q') {
            game_delete(g);
            exit(EXIT_SUCCESS);
        }
        else if (charToInt(choice) >= 0 && charToInt(choice) < NB_COLORS) {
            game_play_one_move(g, (color)charToInt(choice));
            printGame(g);
        }

        // si la partie est fini
        if (game_nb_moves_cur(g) >= game_nb_moves_max(g) && !game_is_over(g)) {
            printf("DOMMAGE\n");
            game_delete(g);
            exit(EXIT_SUCCESS);
        }
        else
            over = game_is_over(g);
    }

    printf("BRAVO\n");
    showCells(g);

    game_delete(g);

    return EXIT_SUCCESS;
}
