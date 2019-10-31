#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>

#include "memoire.h"

static int niveau = 0;
static int trace = 0;

static void
erreur_memoire(char *s)
{
    fprintf(stderr, "Erreur %s mémoire\n", s);
    exit(EXIT_FAILURE);
}

void
memoire_trace(int mode)
{
    trace = mode;
}

static void
tracer(void *p)
{
    printf("[memoire : %d (%04lx)]\n", niveau, ((long)p)&0xffff);
}

void *
memoire_allouer(size_t size)
{
    void *p = malloc(size);
    
    if (p == NULL)
	erreur_memoire("allocation");
    if (trace)
    {
	niveau++;
	tracer(p);
    }

    return p; 
}

void *
memoire_reallouer(void *p, size_t size)
{
    void *q = realloc(p, size);
    
    if (p == NULL && trace)
    {
	niveau++;
	tracer(q);
    }

    if (size != 0 && q == NULL)
	erreur_memoire("reallocation");

    return q;
}

void 
memoire_liberer(void *p)
{
    if (trace)
    {
	niveau--;
	tracer(p);
    }

    free(p);
}
