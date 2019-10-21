#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <game.h>


bool test_game_copy() {
    return true;
}

bool test_game_delete() {
    game g = game_new_empty();
    if (g == NULL) {
        fprintf(stderr, "Error: invalid new empty game!\n");
        return false;
    }
    game_delete(g);

    // TODO : test if game as null
    /* g = NULL;
    game_delete(g); */

    return true;
}

bool test_game_is_over() {
    return true;
}

bool test_game_restart() {
    return true;
}

int main(int argc, char const *argv[])
{
    if (argc == 1) {
        fprintf(stderr, "Usage: %s <testname> [<...>]\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    bool ok = false;

    if (!strcmp(argv[1], "copy"))
        ok = test_game_copy();
    else if (!strcmp(argv[1], "delete"))
        ok = test_game_delete();
    else if (!strcmp(argv[1], "is_over"))
        ok = test_game_is_over();
    else if (!strcmp(argv[1], "restart"))
        ok = test_game_restart();
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
