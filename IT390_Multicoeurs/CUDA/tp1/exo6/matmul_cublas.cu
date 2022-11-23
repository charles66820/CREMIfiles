#include <stdlib.h>
#include <stdio.h>

#include <assert.h>
#include <helper_string.h>  // helper for shared functions common to CUDA SDK samples
#include <timer.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include <cublas_v2.h>

#define TILE_WIDTH 32
#define SIZE 200
/********************** kernel **************************/
__global__
void matmul(float *A, float *B, float *C, int nb_ColA, int nb_ColB, int nb_LigneA, int nb_LigneB)
{

  float tmp_c = 0;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int ligne = blockIdx.y * blockDim.y + threadIdx.y;
  /* m is the Tile ID*/

  for (int k = 0; k < nb_ColB; k++)
    {
      tmp_c += A[ligne * nb_ColA + k] * B[k * nb_LigneB + col];
    }

  C[nb_ColA * ligne + col] = tmp_c; 
  
}

/********************** main **************************/
int main(void)
{
  float *A, *B, *C, *gpu_A, *gpu_B, *gpu_C;
  int nbLigneA, nbLigneB, nbColA, nbColB;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  nbLigneA = TILE_WIDTH * SIZE;
  nbLigneB = TILE_WIDTH * SIZE;
  nbColA = TILE_WIDTH * SIZE;
  nbColB = TILE_WIDTH * SIZE;

  A = (float*) malloc(nbLigneA * nbColA * sizeof(float));
  B = (float*) malloc(nbLigneB * nbColB * sizeof(float));
  C = (float*) malloc(nbLigneA * nbColB * sizeof(float));


  /*Allocation de l'espace pour gpu_x et gpu_y qui vont 
    recevoir x et y sur le GPU*/
  cudaMalloc(&gpu_A, nbLigneA * nbColA * sizeof(float));
  cudaMalloc(&gpu_B, nbLigneB * nbColB * sizeof(float));
  cudaMalloc(&gpu_C, nbLigneA * nbColB * sizeof(float));

  /* Initialisation de x et y*/
  for (int i = 0; i < nbLigneA * nbColA; i++) {
    A[i] = 1.0;
  }


  for (int i = 0; i < nbLigneB * nbColB; i++) {
    B[i] = 2.0;
  }

  /* Copie de A et B sur le GPU dans gpu_x et gpu_y respectivement*/
  cudaMemcpy(gpu_A, A, nbLigneA * nbColA * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_B, B, nbLigneB * nbColB * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_C, C, nbLigneA * nbColB * sizeof(float), cudaMemcpyHostToDevice);

  /* Appel au kernel saxpy sur les N éléments */
  //saxpy<<<(N+255)/256, 256>>>(N, 2.0f, gpu_x, gpu_y);
  //  dim3 grid(N,1);
  dim3 dimGrid((nbColB - 1) / TILE_WIDTH+1, (nbLigneA - 1)/TILE_WIDTH + 1, 1);
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  cudaEventRecord(start);

  // execute the kernel
  int nIter = 1;
  
  // CUBLAS version 2.0 API interface
  {
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    //Perform warmup operation with cublas
    cublasStatus_t ret =
      cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
		  nbColB, nbLigneA, nbColA, &alpha, gpu_B,
		  nbColB, gpu_A, nbColA, &beta,  gpu_C, nbColA);
    //    checkError(ret, "cublas Sgemm returned an error!\n");
    
    // Start Timing (CUBLAS)
    StartTimer();
    
    /* for (int j = 0; j < nIter; j++) */
    /*   { */
	//note cublas is column primary!
	//need to transpose the order
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
		    nbColB, nbLigneA, nbColA, &alpha, gpu_B,
		    nbColB, gpu_A, nbColA, &beta,  gpu_C, nbColA);
	//      }

       double dSeconds = GetTimer()/((double)nIter * 1000.0);
        double dNumOps = 2.0 * (double)nbColA * (double)nbLigneA * (double)nbColB;
        double gflops = 1.0e-9 * dNumOps/dSeconds;

        printf("done.\n");

        //Log througput, etc
        printf("CUBLAS= %.4f GFlop/s, Time= %.2f(ms), Size = %.0f Ops\n",
               gflops, dSeconds*1000., dNumOps);

        // copy result from device to host
        cudaMemcpy(gpu_C, C, nbLigneA * nbColB * sizeof(float), cudaMemcpyDeviceToHost);

        cublasDestroy(handle);
    }
  

  cudaFree(gpu_A);
  cudaFree(gpu_B);
  cudaFree(gpu_C);
  
  free(A);
  free(B);
  free(C);
}
