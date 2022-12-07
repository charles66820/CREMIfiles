#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define NB_THREADS 256

/********************** kernel **************************/
__global__ void inversion(int n, int *x, int *y) {
  /* TODO : Définition de la zone de mémoire partagée pour le block*/
  __shared__ int tmp[256];

  /* TODO : Calcul de l'indice de l'élément dans le tableau initial*/
  int origin = blockIdx.x * blockDim.x + threadIdx.x;
  /* TODO : Calcul de l'indice de l'élément dans le tableau inversé*/
  int destBlock = blockDim.x - 1 - threadIdx.x;
  int dest = (gridDim.x - 1 - blockIdx.x) * blockDim.x + threadIdx.x;

  /* TODO : Ecriture dans la zone de mémoire partagée et dans le tableau*/
  if(origin < n)
    tmp[destBlock] = x[origin];

  __syncthreads();

  if(dest < n)
    y[dest] = tmp[threadIdx.x];
}



/********************** main **************************/
int main(void) {
  int N = NB_THREADS * 1024;
  int i;
  int *x, *y, *gpu_x, *gpu_y;
  x = (int *)malloc(N * sizeof(int));
  y = (int *)malloc(N * sizeof(int));

  /* TODO: Allocation de l'espace pour gpu_x et gpu_y qui vont
    recevoir x et y sur le GPU*/
  cudaMalloc(&gpu_x, N * sizeof(int));
  cudaMalloc(&gpu_y, N * sizeof(int));

  /* Initialisation de x et y*/
  for (int i = 0; i < N; i++) {
    x[i] = i;
  }

  /* TODO : Copie de x et y sur le GPU dans gpu_x et gpu_y respectivement*/
  cudaMemcpy(gpu_x, x, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_y, y, N * sizeof(int), cudaMemcpyHostToDevice);

  /* TODO : Appel au kernel inversion sur les N éléments */
  dim3 grid(1024, 1) ;
  inversion<<<grid, NB_THREADS>>>(N, gpu_x, gpu_y);

  /* TODO : Copie du résultat dans y*/
  cudaMemcpy(y, gpu_y, N * sizeof(int), cudaMemcpyDeviceToHost);

  /* Affichage des 12 premiers éléments*/
  for (i = N - 12; i < N; i++) printf("%d\n", x[i]);
  printf("\n");
  for (i = 0; i < min(12, N); i++) printf("%d\n", y[i]);

  /* TODO : Libération de la mémoire sur le GPU*/
  cudaFree(gpu_x);
  cudaFree(gpu_y);

  free(x);
  free(y);
}
