#include <stdlib.h>
#include <stdio.h>

int main(void)
{
    int* allocatedMemory;

    // On demande d'allouer la memoire
    allocatedMemory = (int*) malloc(sizeof(int));
    // On verifie si la memoire a ete allouee
    if (allocatedMemory == NULL) {
        fprintf(stderr, "not enough memory!\n");
        return EXIT_FAILURE;
    }
    // Utilisation de la memoire allouee
    *allocatedMemory = 10;
    printf("adresse = %p, valeur = %d\n", allocatedMemory, *allocatedMemory);

    // On n'a plus besoin de la memoire, on la libere
    free(allocatedMemory);

    return EXIT_SUCCESS;
}
