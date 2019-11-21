#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "game.h"
//$ clang-format -i test_game_vandrault.c
//tests//

// test of game_new on valid parameters
bool test_game_new(){
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

    game g1 = game_new(cells, 12);
    if (g1==NULL){                  //testing if g1 is a valid pointer
        return false;
    }
    if (game_nb_moves_max(g1)!=12){ //testing if the number of max moves is the same as the one we entered
        return false;
    }
    if (sizeof(cells)!=SIZE*SIZE*sizeof(int)){ //testing if the table is of the right size
        return false;
    }
    for (int i=0; i<SIZE*SIZE; i+=1){ //testing if the cells of the game are the same as the ones of the table
        if (game_cell_current_color(g1,i%12,i/12)!=cells[i]){
            return false;
        }
    }
    game_delete(g1);               //deleting g1 to free the memory
    return true;    
}

//test of game_new_empty
bool test_game_new_empty(){
    game g = game_new_empty();
    if (g==NULL){                  //testing if g1 is a valid pointer
        return false;
    }
    if (game_nb_moves_max(g)!=0){ //testing if the number of max moves is 0
        return false;
    }
    game_play_one_move(g, 0);
    for (int i=0; i<SIZE*SIZE; i++){ //testing if each cell of the game is equal to 0
        if (game_cell_current_color(g,i%12,i/12)!=0){
            return false;
        }
    }
    game_delete(g);               //deleting g to free the memory
    return true;

}

//test of game_set_cell_init on valid parameters
bool test_game_set_cell_init(){
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

    game g1 = game_new(cells, 12);
    if (g1==NULL){                  //testing if g1 is a valid pointer
        return false;
    }
    game_play_one_move(g1, 2);
    game_set_cell_init(g1,5,8,1);
    if (game_cell_current_color(g1,5,8)!=1){ //testing if the color of the cell has been changed
        return false;
    }
    game_delete(g1);               //deleting g1 to free the memory
    return true;
}

//test of game_set_max_moves on valid parameters
bool test_game_set_max_moves(){
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

    game g1 = game_new(cells, 12);
    if (g1==NULL){                  //testing if g1 is a valid pointer
        return false;
    }
    game_set_max_moves(g1,14);
    if (g1==NULL){                  //testing if g1 is still a valid pointer after changing the number of max moves
        return false;
    }
    if (game_nb_moves_max(g1)!=14){ //testing if the number of max moves is changed
        return false;
    }
    game_delete(g1);               //deleting g1 to free the memory
    return true;
}

//main//

int main (int argc,char *argv[]){
    if (argc == 1) {
        fprintf(stderr, "Usage: %s <testname> [<...>]\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    bool ok = false;

    if (!strcmp(argv[1], "new"))
        ok = test_game_new();
    else if (!strcmp(argv[1], "new_empty"))
        ok = test_game_new_empty();
    else if (!strcmp(argv[1], "set_cell_init"))
        ok = test_game_set_cell_init();
    else if (!strcmp(argv[1], "set_max_moves"))
        ok = test_game_set_max_moves();
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