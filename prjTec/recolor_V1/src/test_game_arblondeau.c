#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "game.h"

bool test_game_nb_moves_max() {
    uint nbMax = 12;
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
    game g = game_new(cells, nbMax);
    if (g==NULL){
        fprintf (stderr, "Error : invalid new game \n");
        game_delete(g);
        return false;
    }

    if (game_nb_moves_max(g) != nbMax){
        fprintf (stderr, "Error : nbMaxMoves is not equal to nbMax");
        game_delete(g);
        return false;
    }
    game_delete(g);
    return true;
}

bool test_game_nb_moves_cur(){
    uint nbMax = 12;
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
    game g = game_new(cells, nbMax);
    if (g==NULL){
        fprintf (stderr, "Error : invalid new game \n");
        game_delete(g);
        return false;
    }

    if (game_nb_moves_cur(g) > nbMax){
        fprintf (stderr, "Error : nbCurrentMoves is over to nbMax");
        game_delete(g);
        return false;
    }
    game_delete(g);
    return true;
}

bool test_game_cell_current_color(){
    uint nbMax = 12;
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
    game g = game_new(cells, nbMax);
    if (g==NULL){
        fprintf (stderr, "Error : invalid new game\n");
        game_delete(g);
        return false;
    }

    for (int y = 0; y < SIZE; y++)
    {
        for (int x = 0; x < SIZE; x++)
            printf("%d", game_cell_current_color(g, x, y));
        printf("\n");
    }
    game_delete(g);
    return true;
}

int charToInt(char c) {
    return c - '0';
}


bool test_game_play_one_move(){
    uint nbMax = 12;
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
    game g = game_new(cells, nbMax);
    game gc = game_new(cells, nbMax);
    if (g==NULL || gc == NULL){
        fprintf (stderr, "Error : invalid new game or invalid new game copy \n");
        game_delete(g);
        return false;
    }

    int input = getchar();
    char choice = (char)input;
    if (charToInt(choice) >= 0 && charToInt(choice) < NB_COLORS) {
            game_play_one_move(g, (color)charToInt(choice));
        }
    
    if (g == gc){
        fprintf (stderr, "Error : game_play_one_move does not change the game");
        return false;
    }

    game_delete(g);
    return true;
}


int main (int argc, char const *argv[]){

    if (argc == 1) {
        fprintf(stderr, "Usage: %s <testname> [<...>]\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    bool ok = false;

    if (!strcmp(argv[1], "MaxMoves"))
        ok = test_game_nb_moves_max();
    else if (!strcmp(argv[1], "currentMoves"))
        ok = test_game_nb_moves_cur();
    else if (!strcmp(argv[1], "currentColor"))
        ok = test_game_cell_current_color();
    else if (!strcmp(argv[1], "GamePlayOneMove"))
        ok = test_game_play_one_move();
    else {
        fprintf(stderr, "Error: test \"%s\" not found!\n", argv[1]);
        exit(EXIT_FAILURE);
    }

    // print test result
    if (ok) {
        fprintf(stderr, "Test \"%s\" finished: SUCCESS\n", argv[1]);
        return EXIT_SUCCESS;
    } else {
        fprintf(stderr, "Test \"%s\" finished: FAILURE\n", argv[1]);
        return EXIT_FAILURE;
    }
}
