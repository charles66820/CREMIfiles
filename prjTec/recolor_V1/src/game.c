#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include "game.h"
//#define SIZE 12 //TODO: voire pour le suppriemer

struct game_s{
    color *tab; //le tableau contenant les cases du jeux
    uint nb_moves_max; // nombre de coups max
    uint current_moves; //nombre de coups actuels
    color *tab_init;
    uint size;
};

// enum color_e {
//   RED,
//   GREEN,
//   BLUE,
//   YELLOW,
//   NB_COLORS
// };  // TODO: voire pour le suppriemer

game game_new(color *cells, uint nb_moves_max) {
    if (cells == NULL) {
        fprintf(stderr, "Bad parameter");
        exit(EXIT_FAILURE);
    }

    game g = malloc(sizeof(game));
    if (g == NULL) {
        fprintf(stderr, "Not enough memory");
        exit(EXIT_FAILURE);
    }

    g->nb_moves_max = nb_moves_max;
    g->current_moves = 0;
    g->size = SIZE;

    g->tab = malloc((SIZE * SIZE) * sizeof(color));
    if (g->tab == NULL) {
        fprintf(stderr, "Not enough memory");
        exit(EXIT_FAILURE);
    }

    g->tab_init = malloc((SIZE * SIZE) * sizeof(color));
    if (g->tab_init == NULL) {
        fprintf(stderr, "Not enough memory");
        exit(EXIT_FAILURE);
    }

    for (uint i = 0; i < SIZE * SIZE; i++) {
        g->tab[i] = cells[i];
        g->tab_init[i] = cells[i];
    }

    return g;
}

game game_new_empty(){
    color* tab= (color*) malloc(SIZE*SIZE*sizeof(color));
    for (int i=0; i<SIZE*SIZE; i++){
        tab[i]=0;
    }
    game game_empty = {tab, 0, 0, tab, SIZE};
    return game_empty;
}

void game_set_cell_init(game g, uint x, uint y, color c) {
    if (g == NULL || c >= NB_COLORS || x >= g->size || y >= g->size) {
        fprintf(stderr, "Bad parameter");
        exit(EXIT_FAILURE);
    }

    g->tab[(y * g->size) + x] = c;
    g->tab_init[(y * g->size) + x] = c;
}

void game_set_max_moves(game g, uint nb_max_moves){
    if (g == NULL || nb_max_moves <= 0) exit(EXIT_FAILURE);
    g->nb_moves_max = nb_max_moves;
    return;
}

uint game_nb_moves_max(cgame g){
    if (g == NULL) exit(EXIT_FAILURE);
    return g->nb_moves_max;
}

color game_cell_current_color(cgame g, uint x, uint y){
    if (g == NULL || x>=(g->size) || y>=g->size){
        exit(EXIT_FAILURE);
    }
    return g->tab[x+y*(g->size)];
}

uint game_nb_moves_cur(cgame g){
    if (g == NULL) exit(EXIT_FAILURE);
    return g->current_moves;
}

/**
 * @brief spread color by flood fill algo
 *
 * @param g game object
 * @param x abscissa
 * @param y ordinate
 * @param tc target color
 * @param c color
 */
void ff(game g, uint x, uint y, color tc, color c) {
    if (g == NULL || tc >= NB_COLORS || c >= NB_COLORS) {
        fprintf(stderr, "Bad parameter");
        exit(EXIT_FAILURE);
    }

    if (x >= g->size || y >= g->size || g->tab[(y * g->size) + x] == c) return;
    if (g->tab[(y * g->size) + x] != tc) return;

    g->tab[(y * g->size) + x] = c; // replace target color by color

    ff(g, x + 1, y, tc, c); // spread to right
    ff(g, x + 1, y + 1, tc, c); // spread to right-down
    ff(g, x, y + 1, tc, c); // spread to down
}

void game_play_one_move(game g, color c) {
    if (g == NULL || c >= NB_COLORS) {
        fprintf(stderr, "Bad parameter");
        exit(EXIT_FAILURE);
    }

    ff(g, 0, 0, g->tab[0], c);
}

game game_copy(cgame g){
    if (g == NULL || g->tab==NULL || g->tab_init==NULL){
        exit(EXIT_FAILURE);
    }
    game game_copy;
    for (int i=0; i<g->size*(g->size); i++){
        game_copy->tab[i]=g->tab[i];
        game_copy->tab_init[i]=g->tab_init[i];
    }
    game_copy->nb_moves_max = g->nb_moves_max;
    game_copy->current_moves = g->current_moves;
    game_copy->size = g->size;
}

void game_delete(game g){
    if (g == NULL) exit(EXIT_FAILURE);
    free(g->tab);
    free(g->tab_init);
    free(g);
}

bool game_is_over(cgame g){
    if (g == NULL){
        exit(EXIT_FAILURE);
    }

    color ref = g->tab[0];
    bool over = true;
    for (int i=0 ; i<(g->size)*(g->size) ; i++){
        if (g->tab[i]!=ref){
            over = false;
        }
    }
    return over;
}

void game_restart(game g) {
    if (g == NULL) {
        fprintf(stderr, "Bad parameter");
        exit(EXIT_FAILURE);
    }

    g->current_moves = 0;
    for (uint i = 0; i < g->size * g->size; i++) {
        g->tab[i] = g->tab_init[i];
    }
}