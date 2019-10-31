#include <stddef.h>
#include <assert.h>

#include "memoire.h"
#include "paire.h"

#include "file.h"

struct file
{
    paire tete;
    paire queue;
};

file
file_creer(void)
{
    file self = memoire_allouer(sizeof(struct file));

    self->tete = NULL;
    self->queue = NULL;

    return self;
}

void
file_liberer(file self, void (*liberer_element)(void *))
{
    if (liberer_element != NULL)
	paire_iterer(self->tete, liberer_element);

    while (! file_vide(self))
	file_defiler(self);

    memoire_liberer(self);
}

void
file_enfiler(file self, void *element)
{
    if (file_vide(self))
    {
	self->tete = paire_creer(element, NULL);
	self->queue = self->tete;
    }
    else
    {
	paire nouvelle_paire = paire_creer(element, NULL);
	paire_modifier_cdr(self->queue, nouvelle_paire );
	self->queue = nouvelle_paire;
    }
}

void
file_defiler(file self)
{
    paire paire_suivante;
    assert(! file_vide(self));

    paire_suivante = paire_cdr(self->tete);
    paire_liberer(self->tete);
    self->tete = paire_suivante;
}

void *
file_tete(file self)
{
    assert(! file_vide(self));
    return paire_car(self->tete);
}

int
file_vide(file self)
{
    return self->tete == NULL;
}

void
file_iterer(file self, void (*traitement)())
{
    if (! file_vide(self))
	paire_iterer(self->tete, traitement);
}
