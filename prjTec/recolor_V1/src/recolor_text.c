#include <stdio.h>
#include <stdlib.h>
#include "game.h"


int main(void) {
    bool over = false;
    /*
    game g = game_new_empty();
    game_set_cell_init(g, SIZE, SIZE, RED);
    game_set_max_moves(g, SIZE);//*/

    //*
    int test[2] = {SIZE, SIZE};
    game g = game_new(test, SIZE);//*/

    char choice;
    int input;

    while (!over)
    {
        printf("choisissez une couleur en écrivent un des chiffre suivant :\n 0 = red\n 1 = green\n 2 = blue\n 3 = yellow\n Pour relancer une partie écrivez \"r\"\n Pour quiter écrivez \"q\"\n");
        input = getchar();
        if (input != EOF)
        {
            choice = input;
        }

        if (choice == *"r")
            game_restart(g);
        else if (choice == *"q")
            game_delete(g);
        else if (choice >= 0 && choice < NB_COLORS)
            game_play_one_move(g, (color) choice);
        over = game_is_over(g);
    }

    game_delete(g);
    return EXIT_SUCCESS;
}