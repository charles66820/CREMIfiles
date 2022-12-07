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
  /* TODO : Définition de la zone de mémoire partagée pour le block*/
 
  /* TODO : Calcul de l'indice de l'élément dans le tableau initial*/
 
  /* TODO : Calcul de l'indice de l'élément dans le tableau inversé*/
 
  /* TODO : Ecriture dans la zone de mémoire partagée et dans le tableau*/
  
}

/********************** main **************************/
int main(void)
{
  int N = NB_THREADS * 1024;
  int i;
  int *x, *y, *gpu_x, *gpu_y;
  x = (int*)malloc(N*sizeof(int));
  y = (int*)malloc(N*sizeof(int));

  /* TODO: Allocation de l'espace pour gpu_x et gpu_y qui vont 
    recevoir x et y sur le GPU*/

  /* Initialisation de x et y*/
  for (int i = 0; i < N; i++) {
    x[i] = i;
  }

  /* TODO : Copie de x et y sur le GPU dans gpu_x et gpu_y respectivement*/

  /* TODO : Appel au kernel inversion sur les N éléments */

  /* TODO : Copie du résultat dans y*/


  /* Affichage des 12 premiers éléments*/
  for (i=N-12; i < N; i++)
    printf("%d\n", x[i]);
	   
  for (i = 0; i < min(12, N); i++)
    printf("%d\n", y[i]);

  /* TODO : Libération de la mémoire sur le GPU*/
  
  free(x);
  free(y);
}
