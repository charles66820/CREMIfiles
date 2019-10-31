#include <stdlib.h>
#include <stdio.h>

#include "memoire.h"

#include "paire.h"

struct paire
{
    void *car;
    struct paire *cdr;
};

paire
paire_creer(void *element, paire suivant)
{
    paire self = memoire_allouer(sizeof(struct paire));

    self->car = element;
    self->cdr = suivant;
    return self;
}

void
paire_liberer(paire self)
{
    memoire_liberer(self);
}

void
*paire_car(paire self)
{
    return self->car;
}

paire
paire_cdr(paire self)
{
    return self->cdr;
}

void
paire_modifier_car(paire self, void *element)
{
    self->car = element;
}

void
paire_modifier_cdr(paire self, paire suivant)
{
    self->cdr = suivant;
}

void
paire_iterer(paire self, void (*traitement)())
{
    paire p = self;

    while (p != NULL)
    {
	traitement(p->car);
	p = p->cdr;
    }
}

