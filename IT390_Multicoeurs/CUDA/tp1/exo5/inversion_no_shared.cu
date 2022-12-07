#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define NB_THREADS 256
#define DIM 1024

/********************** kernel **************************/
__global__ void inversion(int n, int *x, int *y) {
  /* TODO : Calcul de l'indice de l'élément dans le tableau initial*/
  int origin = DIM * threadIdx.y + threadIdx.x;
  /* TODO : Calcul de l'indice de l'élément dans le tableau inversé*/
  int dest = DIM * threadIdx.x + threadIdx.y;

  y[dest] = x[origin];
}

/********************** main **************************/
int main(void) {
  int N = NB_THREADS * DIM;
  int i;
  int *x, *y, *gpu_x, *gpu_y;
  x = (int *)malloc(N * sizeof(int));
  y = (int *)malloc(N * sizeof(int));

  /* TOOD : Allocation de l'espace pour gpu_x et gpu_y qui vont
    recevoir x et y sur le GPU*/

  cudaMalloc(&gpu_x, N);
  cudaMalloc(&gpu_y, N);

  /* Initialisation de x et y*/
  for (int i = 0; i < N; i++) {
    x[i] = i;
  }

  /* TODO : Copie de x et y sur le GPU dans gpu_x et gpu_y respectivement*/
  cudaMemcpy(gpu_x, x, N, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_y, y, N, cudaMemcpyHostToDevice);

  /* TODO : Appel au kernel inversion sur les N éléments */
  inversion<<<1, NB_THREADS>>>(N, gpu_x, gpu_y);

  /* TODO : Copie du résultat dans y*/
  cudaMemcpy(y, gpu_y, N, cudaMemcpyDeviceToHost);

  /* Affichage des 12 premiers éléments*/
  for (i = N - 12; i < N; i++) printf("%d\n", x[i]);

  for (i = 0; i < min(12, N); i++) printf("%d\n", y[i]);

  /* TODO : Libération de la mémoire sur le GPU*/
  cudaFree(gpu_x);
  cudaFree(gpu_y);

  free(x);
  free(y);
}
