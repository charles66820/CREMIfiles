#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "solution.h"

bool test_create_solution() {
  uint tab[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  uint length = 16;
  solution sol = create_solution(tab, length);
  if (sol == NULL) {
    fprintf(stderr, "Error : invalid solution.\n");
    return false;
  }
  delete_solution(sol);
  return true;
}

bool test_len_solution() {
  uint tab[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  uint length = 16;
  solution sol = create_solution(tab, length);
  if (sol == NULL) {
    fprintf(stderr, "Error : invalid solution.\n");
    return false;
  }
  if (len_solution(sol) != length) {
    fprintf(stderr, "Error : invalid length for solution.\n");
    return false;
  }
  delete_solution(sol);
  return true;
}

bool test_string_solution() {
  uint tab[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  uint length = 16;
  solution sol = create_solution(tab, length);
  if (sol == NULL) {
    fprintf(stderr, "Error : invalid solution.\n");
    return false;
  }
  char* ssol = string_solution(sol);
  if (strcmp(ssol, "1 2 3 4 5 6 7 8 9 A B C D E F")) {
    fprintf(stderr, "Error : invalid string for solution.\n");
    free(ssol);
    delete_solution(sol);
    return false;
  }
  free(ssol);
  delete_solution(sol);
  return true;
}

bool test_delete_solution() {
  uint tab[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  uint length = 16;
  solution sol = create_solution(tab, length);
  if (sol == NULL) {
    fprintf(stderr, "Error : invalid solution.\n");
    return false;
  }
  delete_solution(sol);
  return true;
}

int main(int argc, char const* argv[]) {
  bool ok = false;

  if (argc == 1) {
    /* fprintf(stderr, "Usage: %s <testname> [<...>]\n", argv[0]);
    exit(EXIT_FAILURE); */

    ok = test_create_solution() && test_len_solution() &&
         test_string_solution() && test_delete_solution();

  } else {
    if (!strcmp(argv[1], "create_solution"))
      ok = test_create_solution();
    else if (!strcmp(argv[1], "len_solution"))
      ok = test_len_solution();
    else if (!strcmp(argv[1], "string_solution"))
      ok = test_string_solution();
    else if (!strcmp(argv[1], "delete_solution"))
      ok = test_delete_solution();
    else {
      fprintf(stderr, "Error: test \"%s\" not found!\n", argv[1]);
      exit(EXIT_FAILURE);
    }
  }

  if (ok) {
    fprintf(stderr, "Test \"%s\" finished: SUCCESS\n", argv[1]);
    return EXIT_SUCCESS;
  } else {
    fprintf(stderr, "Test \"%s\" finished: FAILURE\n", argv[1]);
    return EXIT_FAILURE;
  }
}