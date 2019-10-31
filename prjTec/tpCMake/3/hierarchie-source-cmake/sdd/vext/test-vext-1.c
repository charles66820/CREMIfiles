#include <stdlib.h>
#include <stdio.h>

#include "memoire.h"
#include "chaine.h"

#include "vext.h"

static void
afficher_element(void *element)
{
    printf("%s ", (char *) element);
}

static vext
faire_v1(void)
{
    vext v1 = vext_creer();

    vext_ajouter(v1, "10");
    vext_ajouter(v1, "20");
    vext_ajouter(v1, "30");
    vext_ecrire(v1, 1, "40");

    return v1;
}

int
main(int argc, char *argv[])
{
    vext v1 = faire_v1();

    printf("v1 : ");
    vext_iterer(v1, afficher_element);
    printf("\n");

    vext_liberer(v1);

    return EXIT_SUCCESS;
}

