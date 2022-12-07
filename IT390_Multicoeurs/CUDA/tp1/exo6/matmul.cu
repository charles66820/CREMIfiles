#include <stdlib.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#define TILE_WIDTH 32
#define SIZE 20
/********************** kernel **************************/
__global__
void matmul(float *A, float *B, float *C, int nb_ColA, int nb_ColB, int nb_LigneA, int nb_LigneB)
{
  
}

/********************** main **************************/
int main(void)
{
  float *A, *B, *C, *gpu_A, *gpu_B, *gpu_C;
  int nbLigneA, nbLigneB, nbColA, nbColB;

  
  nbLigneA = TILE_WIDTH * SIZE;
  nbLigneB = TILE_WIDTH * SIZE;
  nbColA = TILE_WIDTH * SIZE;
  nbColB = TILE_WIDTH * SIZE;

  A = (float*) malloc(nbLigneA * nbColA * sizeof(float));
  B = (float*) malloc(nbLigneB * nbColB * sizeof(float));
  C = (float*) malloc(nbLigneA * nbColB * sizeof(float));

  /*Allocation de l'espace pour le GPU */

  /* Initialisation de A et B*/
  for (int i = 0; i < nbLigneA * nbColA; i++) {
    A[i] = 1.0;
  }


  for (int i = 0; i < nbLigneB * nbColB; i++) {
    B[i] = 2.0;
  }

  /* Copie de A et B sur le GPU */

  /* Lancement du kernel avec mesure du temps */

  /* Vérification du résultat*/
  float maxError = 0.0f;
  for (int i = 0; i < nbLigneA * nbColB; i++)
    {
      maxError = max(maxError, abs(C[i]- 2*nbLigneB));
    }
  printf("Max error: %f\n", maxError);

  /* Libération de la mémoire sur le GPU*/
  
  free(A);
  free(B);
  free(C);
}
