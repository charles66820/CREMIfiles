#include <stdlib.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

void initArrays(float* x, float* y,int N){
  for(int i=0;i<N;i++){
    x[i] = 1;
    y[i] = 2;
  }
}

__global__
void addArrays(float*x,float* y,int N){
  
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(int i=index;i<N;i+=stride)
    y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 1<<20;
  float *x, *y;
  
  // Allocate memory on the GPU that is also accessible on the host
  cudaMallocManaged(&x,N*sizeof(float));
  cudaMallocManaged(&y,N*sizeof(float));
  
  initArrays(x,y,N);
  
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  addArrays<<<numBlocks,blockSize>>>(x,y,N);
  
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  
  cudaFree(x);
  cudaFree(y);
  
}
