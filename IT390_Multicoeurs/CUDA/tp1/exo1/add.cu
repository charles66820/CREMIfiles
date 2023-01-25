#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <random>

#define N (1024 * 1024)

// __global__ void kernel(void) {}

__global__ void add(int *a, int *b, int *c, int n) {
  int indice = threadIdx.x + blockIdx.x * blockDim.x;

  if (indice < n) c[indice] = a[indice] + b[indice];
}

void random_ints(int *list, int size) {
  for (size_t i = 0; i < size; i++) {
    list[i] = rand() % 100;
  }
}

int main(void) {
  int *a, *b, *c;
  int *gpu_a, *gpu_b, *gpu_c;
  int size = N * sizeof(int);

  cudaMalloc((void **)&gpu_a, size);
  cudaMalloc((void **)&gpu_b, size);
  cudaMalloc((void **)&gpu_c, size);

  a = (int *)malloc(size);
  b = (int *)malloc(size);
  c = (int *)malloc(size);
  random_ints(a, N);
  random_ints(b, N);

  cudaMemcpy(gpu_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_b, b, size, cudaMemcpyHostToDevice);

  add<<<1024, 1024>>>(gpu_a, gpu_b, gpu_c, N);

  cudaMemcpy(c, gpu_c, size, cudaMemcpyDeviceToHost);

  cudaFree(gpu_a);
  cudaFree(gpu_b);
  cudaFree(gpu_c);

  for (int i = 0; i < N; i++) {
    printf("%d %d %d\n", a[i], b[i], c[i]);
  }

  free(a);
  free(b);
  free(c);
  return 0;
}