#include <stdlib.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

/********************** kernel **************************/
__global__
void saxpy(int n, float a, float *x, float *y)
{
  /* Calcul de l'indice i*/
  //  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int i = blockIdx.x;
  /* Calcul de saxpy*/
  //if (i < n) y[i] = a*x[i] + y[i];
  if(i < n)
    for(int j = 0; j < n; j++)
      y[i] = a*x[i] + y[i];
}

/********************** main **************************/
int main(void)
{
  int N = 1<<20;
  float *x, *y, *gpu_x, *gpu_y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  /*Allocation de l'espace pour gpu_x et gpu_y qui vont 
    recevoir x et y sur le GPU*/
  cudaMalloc(&gpu_x, N*sizeof(float)); 
  cudaMalloc(&gpu_y, N*sizeof(float));

  /* Initialisation de x et y*/
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  /* Copie de x et y sur le GPU dans gpu_x et gpu_y respectivement*/

  /* Appel au kernel saxpy sur les N éléments */
  //saxpy<<<(N+255)/256, 256>>>(N, 2.0f, gpu_x, gpu_y);
  //  dim3 grid(N,1);
  cudaStream_t stream1, stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);

  // cudaMemcpy(gpu_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(gpu_x, x, N*sizeof(float), cudaMemcpyHostToDevice, stream1);

  for(int i = 0; i < 10; i ++){
    // cudaMemcpy(gpu_y, y, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(gpu_y, y, N*sizeof(float), cudaMemcpyHostToDevice, stream1);
    saxpy<<<1, 1, 0, stream1>>>(N, 2.0f, gpu_x, gpu_y);
    cudaDeviceSynchronize();
  }

  /* Copie du résultat dans y*/
  cudaMemcpy(y, gpu_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    {
      maxError = max(maxError, abs(y[i]-4.0f));
    }
  printf("Max error: %f\n", maxError);

  /* Libération de la mémoire sur le GPU*/
  cudaFree(gpu_x);
  cudaFree(gpu_y);
  
  free(x);
  free(y);
}
