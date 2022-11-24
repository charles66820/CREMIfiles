#include <stdlib.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#define NB_THREADS 256

/********************** kernel **************************/
__global__
void inversion(int n, int *x, int *y)
{
  /* Définition de la zone de mémoire partagée pour le block*/
  __shared__ int tmp[NB_THREADS];
  /* Calcul de l'indice de l'élément dans le tableau initial*/
  int origin = blockIdx.x * blockDim.x + threadIdx.x;
  /* Calcul de l'indice de l'élément dans le tableau inversé*/
  int tmp_dest = blockDim.x - 1 - threadIdx.x;
  int dest = (gridDim.x - 1 - blockIdx.x) * blockDim.x + threadIdx.x;
  //if (i < n) y[i] = a*x[i] + y[i];
  if( origin < n)
    tmp[tmp_dest] = x[origin];
  __syncthreads();

  if(dest < n)
    y[dest] = tmp[threadIdx.x];
  
}

/********************** main **************************/
int main(void)
{
  int N = NB_THREADS * 1024;
  int i;
  int *x, *y, *gpu_x, *gpu_y;
  x = (int*)malloc(N*sizeof(int));
  y = (int*)malloc(N*sizeof(int));

  /*Allocation de l'espace pour gpu_x et gpu_y qui vont 
    recevoir x et y sur le GPU*/
  cudaMalloc((void**)&gpu_x, N*sizeof(int));
  cudaMalloc((void**)&gpu_y, N*sizeof(int)); 

  /* Initialisation de x et y*/
  for (int i = 0; i < N; i++) {
    x[i] = i;
  }

  /* Copie de x et y sur le GPU dans gpu_x et gpu_y respectivement*/
  cudaMemcpy(gpu_x, x, N*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_y, y, N*sizeof(int), cudaMemcpyHostToDevice);

  /* Appel au kernel saxpy sur les N éléments */
  //saxpy<<<(N+255)/256, 256>>>(N, 2.0f, gpu_x, gpu_y);
  //  dim3 grid(N,1);
  inversion<<<(N + NB_THREADS - 1)/NB_THREADS, NB_THREADS>>>(N, gpu_x, gpu_y);

  /* Copie du résultat dans y*/
  cudaMemcpy(y, gpu_y, N*sizeof(int), cudaMemcpyDeviceToHost);

  /* Affichage des 12 premiers éléments*/
  for (i=N-12; i < N; i++)
    printf("%d\n", x[i]);
	   
  for (i = 0; i < min(12, N); i++)
    printf("%d\n", y[i]);

  /* Libération de la mémoire sur le GPU*/
  cudaFree(gpu_x);
  cudaFree(gpu_y);
  
  free(x);
  free(y);
}
