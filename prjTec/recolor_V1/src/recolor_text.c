#include <stdio.h>
#include <stdlib.h>
#include "game.h"

void showCells(game g)
{
    for (int x = 0; x < SIZE; x++)
    {
        for (int y = 0; y < SIZE; y++)
            printf("%d", game_cell_current_color(g, x, y));
        printf("\n");
    }
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

    game g = game_new(cells, 12);

    while (!over)
    {
        printf("\n");
        printf("nb coups joués: %d ; nb coups max : %d\n", game_nb_moves_cur(g), nbMaxHit);

        showCells(g); // affiche la grille

        printf("Jouer un coup: (0,1,2,3,r ou q ; r pour redémarrer ou q pour quitter)\n");

        int input;
        char choice;

        while ((input = getchar()) != '\n' && input != EOF)
        {
            choice = (char)input;
        }

        // client inputs
        if (choice == 'r')
            game_restart(g);
        else if (choice == 'q') {
            game_delete(g);
            exit(EXIT_SUCCESS);
        }
        else if (choice - '0' >= 0 && choice - '0' < NB_COLORS)
        {
            game_play_one_move(g, (color)choice - '0');
        }

        // si la partie est fini
        if (game_nb_moves_cur(g) > game_nb_moves_max(g)) {
            printf("DOMMAGE\n");
            exit(EXIT_SUCCESS);
        }
        else
            over = game_is_over(g);
    }

    printf("BRAVO\n");

    return EXIT_SUCCESS;
}
