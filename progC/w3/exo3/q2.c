#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "barcode.h"

void freeBarCode(struct barCode b)
{
    free(b.code);
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
    printBarCode(b, 10);               // given function
    freeBarCode(b);                    // given function
    return EXIT_SUCCESS;
}