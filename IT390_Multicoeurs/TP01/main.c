#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define SIZE 10000000

typedef float type;

// gcc -o main main.c -lm

void dep(type* x, type* A) {
  for (size_t j = 0; j < 100; j++)
    for (size_t i = 0; i < SIZE; i++) {
      *x += 1. / sqrt(A[i]);
    }
}

int main(int argc, char const* argv[]) {
  type* A = (type*)malloc(SIZE * sizeof(double));
  if (A == NULL) exit(EXIT_FAILURE);

  for (size_t i = 0; i < SIZE; i++) A[i] = i + 1;

  type x = 0;

  dep(&x, A);

  printf("%f\n", x);

  free(A);

  return EXIT_SUCCESS;
}
