#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <stdio.h>
#include <stdlib.h>

/********************** kernel **************************/
__global__ void saxpy(int n, float a, float *x, float *y) {
  /* TODO : Calcul de l'indice i*/
  int i = blockIdx.x;

  /* Calcul de saxpy*/
  if (i < n) y[i] = a * x[i] + y[i];
}

/********************** main **************************/
int main(void) {
  int N = 1 << 20;
  float *x, *y, *gpu_x, *gpu_y;
  x = (float *)malloc(N * sizeof(float));
  y = (float *)malloc(N * sizeof(float));

  /* TODO : Allocation de l'espace pour gpu_x et gpu_y qui
     vont recevoir x et y sur le GPU*/
  cudaMalloc(&gpu_x, N * sizeof(float));
  cudaMalloc(&gpu_y, N * sizeof(float));

  /* Initialisation de x et y*/
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  /* TODO : Copie de x et y sur le GPU dans gpu_x et gpu_y respectivement*/
  cudaMemcpy(gpu_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

  /* TODO : Appel au kernel saxpy sur les N éléments avec a=2.0f */
  saxpy<<<N, 1>>>(N, 2.0f, gpu_x, gpu_y);

  /* TODO : Copie du résultat dans y*/
  cudaMemcpy(y, gpu_y, N * sizeof(float), cudaMemcpyDeviceToHost);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++) {
    if (y[i] != 4.0f) printf("not equal %d %f %f\n", i, y[i], x[i]);
    maxError = max(maxError, abs(y[i] - 4.0f));
  }
  printf("Max error: %f\n", maxError);

  /* TODO : Libération de la mémoire sur le GPU*/
  cudaFree(gpu_x);
  cudaFree(gpu_y);

  free(x);
  free(y);
}
