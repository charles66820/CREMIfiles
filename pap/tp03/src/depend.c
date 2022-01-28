#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define T 10
int A[T][T];

int k = 0;

void tache(int i, int j) {
  volatile int x = random() % 1000000;
  for (int z = 0; z < x; z++)
    ;
#pragma omp atomic capture
  A[i][j] = k++;
}

int main(int argc, char **argv) {
  int i, j;

  // génération des taches
#pragma omp parallel
#pragma omp single
  for (i = 0; i < T; i++)
    for (j = 0; j < T; j++)
#pragma omp task firstprivate(i, j)
      tache(i, j);

  // affichage du tableau
  for (i = 0; i < T; i++) {
    puts("");
    for (j = 0; j < T; j++) printf(" %2d ", A[i][j]);
  }

  return 0;
}
