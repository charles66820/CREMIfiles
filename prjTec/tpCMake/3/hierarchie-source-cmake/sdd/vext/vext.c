#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

#include "memoire.h"

#include "vext.h"

#define TAILLE_BLOC 4

struct vext 
{
    void **vecteur;
    unsigned int taille_physique;
    unsigned int taille_rallongement;
    unsigned int nombre_d_elements;
    void (*liberer)(void *);
    void (*afficher)(void *);
};

static void
L_afficher(void *p)
{
    printf("<%p> ", p);
}

vext
vext_creer(void)
{
    vext self = memoire_allouer(sizeof(struct vext));

    self->vecteur = NULL;
    self->taille_physique = 0;
    self->taille_rallongement = 1;
    self->nombre_d_elements = 0;
    self->liberer = NULL;
    self->afficher = L_afficher;

    return self;
}

void
vext_liberer(vext self)
{
    void **p = self->vecteur;  
    unsigned int n = self->nombre_d_elements; 
 
    if (self->liberer != NULL)
    {
	while (n > 0)  
	{
	    if (*p != NULL) 
		self->liberer(*p); 
	    p++; 
	    n--; 
	}
    }
    memoire_liberer(self->vecteur);
    memoire_liberer(self);
}

static void
rallonger(vext self, unsigned int nouvelle_taille)
{
    self->vecteur = memoire_reallouer(self->vecteur,
				      nouvelle_taille*sizeof(void *));
    memset(self->vecteur + self->taille_physique, 
	   0, 
	   nouvelle_taille - self->taille_physique);
    self->taille_physique = nouvelle_taille;
}

static unsigned int
nouvelle_taille(vext self, unsigned int i)
{
    unsigned int n = self->taille_physique;

    
    while (i >= n)
    {
	n += self->taille_rallongement;
	self->taille_rallongement *= 2;
    }
    return n;
}

void
vext_ecrire(vext self, unsigned int i, void *valeur)
{
    assert(i >= 0);

    if (i >= self->taille_physique)
	rallonger(self, nouvelle_taille(self, i));

    self->vecteur[i] = valeur;
    if (self->nombre_d_elements < i+1)
	self->nombre_d_elements = i+1;
}

void *
vext_lire(vext self, unsigned int i)
{
    assert(i >= 0 && i < self->nombre_d_elements);

    return self->vecteur[i];
}


void 
vext_ajouter(vext self, void *valeur)
{
    vext_ecrire(self, self->nombre_d_elements, valeur);
}

  
unsigned int
vext_nombre_d_elements(vext self)
{
    return self->nombre_d_elements;
}

void
vext_iterer(vext self, void (*fonction)(void *))
{
    unsigned int i = 0;
    unsigned int n = self->nombre_d_elements;
    void **v = self->vecteur;

    while (i < n)
    {
	fonction(v[i]);
	i++;
    }
}

void
vext_definir_liberation(vext self, void (*liberer)(void *))
{
    self->liberer = liberer;
}

void
vext_definir_affichage(vext self, void (*afficher)(void *))
{
    self->afficher = afficher;
}

void
vext_afficher(vext self)
{
    if (self->afficher != NULL)
    {
	printf("[ ");
	vext_iterer(self, self->afficher);
	printf("] ");
    }
    else
	printf("[...] ");
}
