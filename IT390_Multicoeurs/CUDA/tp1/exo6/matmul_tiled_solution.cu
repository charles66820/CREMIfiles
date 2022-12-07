#include <stdlib.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#define TILE_WIDTH 16
#define SIZE 20
/********************** kernel **************************/
__global__
void matmul(float *A, float *B, float *C, int nb_ColA, int nb_ColB, int nb_LigneA, int nb_LigneB)
{
  __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH];
  __shared__ float tile_B[TILE_WIDTH][TILE_WIDTH];

  float tmp_c = 0;
  int indice_A, indice_B;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int ligne = blockIdx.y * blockDim.y + threadIdx.y;
  /* m is the Tile ID*/
  for(int m = 0; m < (nb_ColA - 1)/TILE_WIDTH + 1; m++)
    {
      /* Vérifier si les indices des threads sont dans la matrice dans le cas 
	 où la dimension des blocks est plus grande que la matrice*/
      if ( (ligne < nb_LigneA) && (TILE_WIDTH * m + threadIdx.x < nb_ColA) )
	{
	  indice_A = ligne * nb_ColA + TILE_WIDTH * m + threadIdx.x;
	  tile_A[threadIdx.y][threadIdx.x] = A[indice_A];
	}
      else
	tile_A[threadIdx.y][threadIdx.x] = 0;
      if ((col < nb_ColB) && ( m * TILE_WIDTH  + threadIdx.y < nb_LigneB))
	{
	  indice_B = col + nb_ColB * TILE_WIDTH * m + threadIdx.y * nb_ColB;
	  tile_B[threadIdx.y][threadIdx.x] = B[indice_B];
	}
      else
	tile_B[threadIdx.y][threadIdx.x] = 0;
      __syncthreads();
      /* Les deux tuiles sont copiées en mémoire, on peut faire le calcul partiel de C*/
      for (int k = 0; k < TILE_WIDTH; k ++)
	{
	  tmp_c += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
	}
      __syncthreads();
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

  printf("%d\n", sizeof(float));
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
  matmul<<<dimGrid, dimBlock>>>(gpu_A, gpu_B, gpu_C, nbColA, nbColB, nbLigneA, nbLigneB);
  cudaEventRecord(stop);
  
  /* Copie du résultat dans y*/
  cudaMemcpy(C, gpu_C, nbLigneA * nbColB * sizeof(float), cudaMemcpyDeviceToHost);
  
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("it took %f ms to execute \n", milliseconds);
  /* Affichage des 12 premiers éléments*/
  float maxError = 0.0f;
  for (int i = 0; i < nbLigneA * nbColB; i++)
    {
      maxError = max(maxError, abs(C[i]- 2*nbLigneB));
    }
  printf("Max error: %f\n", maxError);

  /* Libération de la mémoire sur le GPU*/
  cudaFree(gpu_A);
  cudaFree(gpu_B);
  cudaFree(gpu_C);
  
  free(A);
  free(B);
  free(C);
}
