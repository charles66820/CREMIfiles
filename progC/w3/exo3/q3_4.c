#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "barcode.h"

struct barCode generateBarCode(unsigned int s)
{
    struct barCode bc; // déclare la structure bc

    if (s < 3)
        exit(EXIT_FAILURE);

    bc.size = s;

    // On demande d'allouer la memoire pour le tableau code de la stucture bc
    bc.code = malloc(s * sizeof(int));
    // On verifie si la memoire a ete allouee
    if (bc.code == NULL)
    {
        fprintf(stderr, "not enough memory!\n");
        exit(EXIT_FAILURE);
    }

    // mette un nommer de 1 a 5 dans le code
    for (int i = 0; i < s; i++)
    {
        bc.code[i] = rand() % 4 + 1;
    }

    return bc;
}

void freeBarCode(struct barCode b)
{
    free(b.code);
}

void printLine(struct barCode b)
{
    if (b.size < 3)
        exit(EXIT_FAILURE);

    if (b.code == NULL)
        exit(EXIT_FAILURE);

    for (int i = 0; i < b.size; i++)
        for (int j = 0; j < b.code[i]; j++)
            if (i % 2)
                printf(" ");
            else
                printf("*");
    printf("\n");
}

void printBarCode(struct barCode b, unsigned int h)
{
    if (h < 1)
        exit(EXIT_FAILURE);

    for (int i = 0; i <= h; i++)
        printLine(b);
}

/** This main function is given as a sample and can
 * be changed as needed for debugging purpose
 **/
int main(int argc, char *argv[])
{
    struct barCode b;
    int seed = time(NULL);
    srand(seed);                         //initialize the pseudo random generator seed
    unsigned int randSize = rand() % 10; // generate a random number between 0 and 9;
    printf("Generating a bar code of size %u\n", randSize + 1);
    b = generateBarCode(randSize + 1); // function to be defined
    printBarCode(b, 10); // given function
    freeBarCode(b);      // given function
    return EXIT_SUCCESS;
}