#include <stdio.h>
#include <stdlib.h>
#include "game.h"

int main(void) {
    game g = game_new_empty();//game g = game_new(RED, 12);
    game_delete(g);
    return EXIT_SUCCESS;
}