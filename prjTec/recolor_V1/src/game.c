#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#define SIZE 12

typedef struct game_s{
    color *tab; //le tableau contenant les cases du jeux (2D ?)
    unsigned int nb_moves_max; // nombre de coups max
    int current_moves; //nombre de coups actuels
    color *tab_init;
    unsigned int size;
}*game;

typedef enum color_e {RED, GREEN, BLUE, YELLOW, NB_COLORS} color;


//jeux en mémoire
typedef const struct game_s{
    /*color *tab; //le tableau contenant les cases du jeux
    unsigned int nb_moves_max; // nombre de coups max
    int current_moves; //nombre de coups actuels*/
}*cgame;

game game_new(color *cells, unsigned int nb_moves_max){return;}

game game_new_empty(){return;}

void game_set_cell_init(game g, unsigned int x, unsigned int y, color c){}

void game_set_max_moves(game g, unsigned int nb_max_moves){}

unsigned int game_nb_moves_max(cgame g){return 0;}

color game_cell_current_color(cgame g, unsigned int x, unsigned int y){return BLUE;}

unsigned int game_nb_moves_cur(cgame g){return 0;}

void game_play_one_move(game g, color c){} //flood fill algo

game game_copy(cgame g){return;}

void game_delete(game g){}

bool game_is_over(cgame g){return true;}

void game_restart(game g){} 