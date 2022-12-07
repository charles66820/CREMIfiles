#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define TILE_WIDTH 32
#define SIZE 20

int main(void) {
  float *A, *B, *C;
  int nbLigneA, nbLigneB, nbColA, nbColB;

  nbLigneA = TILE_WIDTH * SIZE;
  nbLigneB = TILE_WIDTH * SIZE;
  nbColA = TILE_WIDTH * SIZE;
  nbColB = TILE_WIDTH * SIZE;

  A = (float *)malloc(nbLigneA * nbColA * sizeof(float));
  B = (float *)malloc(nbLigneB * nbColB * sizeof(float));
  C = (float *)malloc(nbLigneA * nbColB * sizeof(float));

  /* Initialisation de x et y*/
  for (int i = 0; i < nbLigneA * nbColA; i++) {
    A[i] = 1.0;
  }

  for (int i = 0; i < nbLigneB * nbColB; i++) {
    B[i] = 2.0;
  }

  for (int i = 0; i < nbLigneB * nbColB; i++) {
    C[i] = 0.0;
  }

  struct timeval tv;
  struct timeval start_tv;

  gettimeofday(&start_tv, NULL);

#pragma omp parallel for
  for (int i = 0; i < nbLigneA; i++)
    for (int j = 0; j < nbColB; j++)
      for (int k = 0; k < nbColA; k++)
        C[i * nbColA + j] =
            C[i * nbColA + j] + A[i * nbColA + k] * B[k + nbLigneB * j];

  gettimeofday(&tv, NULL);
  double elapsed = ((tv.tv_sec - start_tv.tv_sec) +
                    (tv.tv_usec - start_tv.tv_usec) / 1000000.0) *
                   1000;

  printf("it took %f ms to execute \n", elapsed);
  free(A);
  free(B);
  free(C);
}
