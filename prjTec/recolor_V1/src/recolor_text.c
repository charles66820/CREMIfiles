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
    int nbMaxHit = 4;

    //*
    game g = game_new_empty(); // crée une partie vide
    game_set_max_moves(g, nbMaxHit); // iniciamise le nombre de coup max
    game_set_cell_init(g, 11, 11, GREEN); // defini la coule d'une case  */

    /* crée une partie a partire d'a tableu
    int cells[12][12] = {{0, 2, 3, 1, 2, 3, 2, 3, 1, 0, 2, 3}, {0, 2, 3, 1, 2, 3, 2, 3, 1, 0, 2, 3}, {0, 2, 3, 1, 2, 3, 2, 3, 1, 0, 2, 3}, {0, 2, 3, 1, 2, 3, 2, 3, 1, 0, 2, 3}, {0, 2, 3, 1, 2, 3, 2, 3, 1, 0, 2, 3}, {0, 2, 3, 1, 2, 3, 2, 3, 1, 0, 2, 3}, {0, 2, 3, 1, 2, 3, 2, 3, 1, 0, 2, 3}, {0, 2, 3, 1, 2, 3, 2, 3, 1, 0, 2, 3}, {0, 2, 3, 1, 2, 3, 2, 3, 1, 0, 2, 3}, {0, 2, 3, 1, 2, 3, 2, 3, 1, 0, 2, 3}, {0, 2, 3, 1, 2, 3, 2, 3, 1, 0, 2, 3}, {0, 2, 3, 1, 2, 3, 2, 3, 1, 0, 2, 3}};
    game g = game_new(cells, nbMaxHit);//*/

    char choice;
    int input;

    while (!over)
    {
        printf("nb coups joués: %d ; nb coups max : %d\n", game_nb_moves_cur(g), nbMaxHit);

        showCells(g); // affiche la grille

        printf("Jouer un coup: (0,1,2,3,r ou q ; r pour redémarrer ou q pour quitter)\n");
        input = getchar();
        if (input != EOF)
        {
            choice = (char) input;
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
            printf("DOMMAGE");
            exit(EXIT_SUCCESS);
        }
        else
            over = game_is_over(g);
    }

    printf("BRAVO");

    return EXIT_SUCCESS;
}
