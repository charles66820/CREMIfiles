#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_WIDTH 32
#define SIZE 20
/********************** kernel **************************/
__global__ void matmul(float *A, float *B, float *C, int nb_ColA, int nb_ColB,
                       int nb_LigneA, int nb_LigneB) {
//TODO:
}

/********************** main **************************/
int main(void) {
  float *A, *B, *C, *gpu_A, *gpu_B, *gpu_C;
  int nbLigneA, nbLigneB, nbColA, nbColB;

  nbLigneA = TILE_WIDTH * SIZE;
  nbLigneB = TILE_WIDTH * SIZE;
  nbColA = TILE_WIDTH * SIZE;
  nbColB = TILE_WIDTH * SIZE;

  int sizeA = nbLigneA * nbColA * sizeof(float);
  int sizeB = nbLigneB * nbColB * sizeof(float);
  int sizeC = nbLigneA * nbColB * sizeof(float);

  A = (float *)malloc(sizeA);
  B = (float *)malloc(sizeB);
  C = (float *)malloc(sizeC);

  /*Allocation de l'espace pour le GPU */
  cudaMalloc(&gpu_A, sizeA);
  cudaMalloc(&gpu_B, sizeB);
  cudaMalloc(&gpu_C, sizeC);

  /* Initialisation de A et B*/
  for (int i = 0; i < nbLigneA * nbColA; i++) {
    A[i] = 1.0;
  }

  for (int i = 0; i < nbLigneB * nbColB; i++) {
    B[i] = 2.0;
  }

  /* Copie de A et B sur le GPU */
  cudaMemcpy(gpu_A, A, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_A, B, sizeB, cudaMemcpyHostToDevice);

  /* Lancement du kernel avec mesure du temps */
  dim3 grid(TILE_WIDTH, TILE_WIDTH);
  dim3 block(TILE_WIDTH, TILE_WIDTH);
  matmul<<<grid, block>>>(gpu_A, gpu_B, gpu_C, nbColA, nbColB, nbLigneA, nbLigneB);

  cudaMemcpy(C, gpu_C, sizeC, cudaMemcpyDeviceToHost);

  /* Vérification du résultat*/
  float maxError = 0.0f;
  for (int i = 0; i < nbLigneA * nbColB; i++) {
    maxError = max(maxError, abs(C[i] - 2 * nbLigneB));
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
