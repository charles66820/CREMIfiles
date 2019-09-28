#include <stdlib.h>
#include <stdio.h>

void usage(char *cmd)
{
    fprintf(stderr, "%s <integer>... \n", cmd);
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[])
{
    if (argc == 1)
    {
        usage(argv[0]);
    }

    int *tab;

    // On demande d'allouer la memoire pour le tableau
    tab = (int *)malloc((argc - 1) * sizeof(int));
    // On verifie si la memoire a ete allouee
    if (tab == NULL)
    {
        fprintf(stderr, "not enough memory!\n");
        return EXIT_FAILURE;
    }

    // Utilisation de la memoire en parten des dernier argument jusqu'au premier
    for (int i = 0; i < argc; i = i + 1)
    {
        tab[i] = atoi(argv[argc - (i + 1)]);
    }

    for (int i = 0; i < argc-1; i = i + 1)
    {
        printf("tab[%d] = %d\n", i, tab[i]);
    }

    // On n'a plus besoin de la memoire, on la libere
    free(tab);

    return EXIT_SUCCESS;
}
