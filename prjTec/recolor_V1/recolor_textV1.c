#include <stdio.h>
#include <stdlib.h>
#include <time.h>
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

void initGameCells(game g)
{
    for (int x = 0; x < SIZE; x++)
    {
        for (int y = 0; y < SIZE; y++)
        {
            int color = rand()%NB_COLORS;
            game_set_cell_init(g, x, y, color); // defini la coule d'une case
        }
    }
}

int main(void) {
    bool over = false;

    int nbMaxHit = 12;

    srand(time(NULL));

    game g = game_new_empty();       // crée une partie vide
    game_set_max_moves(g, nbMaxHit); // iniciamise le nombre de coup max
    initGameCells(g);                // inicialise la grille

    char choice;
    int input;

    while (!over)
    {
        printf("\n");
        printf("nb coups joués: %d ; nb coups max : %d\n", game_nb_moves_cur(g), nbMaxHit);

        showCells(g); // affiche la grille

        printf("Jouer un coup: (0,1,2,3,r ou q ; r pour redémarrer ou q pour quitter)\n");
        input = getchar();
        if (input != EOF)
        {
            choice = (char) input;
        }

        // client inputs
        if (choice == 'r') {
            game_restart(g);
            initGameCells(g); // inicialise la grille
        }
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
