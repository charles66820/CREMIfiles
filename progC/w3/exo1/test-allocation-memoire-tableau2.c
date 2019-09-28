#include <stdlib.h>
#include <stdio.h>

void usage(char * cmd) {
    fprintf(stderr,"%s <integer array size> \n", cmd);
    exit(EXIT_FAILURE);
}

int main(int argc, char*argv[]) {
    if (argc != 2){
        usage(argv[0]);
    }
    int* tab;
    int size = atoi(argv[1]);

    // On demande d'allouer la memoire pour le tableau
    tab = (int*) malloc(size * sizeof(int));
    // On verifie si la memoire a ete allouee
    if (tab == NULL) {
        fprintf(stderr, "not enough memory!\n");
        return EXIT_FAILURE;
    }

    // Utilisation de la memoire
    for (int i=0; i<size; i=i+1){
        tab[i] = i;
    }
    for (int i=0; i<size; i=i+1){
        printf("tab[%d] = %d\n", i, tab[i]);
    }

    // On demande d'allouer la memoire pour le tableau 2
    int *tab2 = (int *)malloc(size * sizeof(int));
    // On verifie si la memoire a ete allouee
    if (tab == NULL)
    {
        fprintf(stderr, "not enough memory!\n");
        return EXIT_FAILURE;
    }

    // On copie les valeurs de tab a tab2
    for (int i = 0; i < size; i = i + 1)
    {
        tab2[i] = tab[i];
    }

    // On n'a plus besoin de la memoire, on la libere
    free(tab);

    // On affiche le contenu du tableau 2
    for (int i = 0; i < size; i = i + 1)
    {
        printf("tab2[%d] = %d\n", i, tab2[i]);
    }

    return EXIT_SUCCESS;
}
