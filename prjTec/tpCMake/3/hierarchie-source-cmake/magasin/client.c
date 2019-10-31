#include <stdio.h>

#include "memoire.h"
#include "chaine.h"

#include "client.h"

struct client
{
    char *nom;
    int nombre_d_articles;
};

client
client_creer(char *nom, int nombre_d_articles)
{
    client self = memoire_allouer(sizeof(struct client));

    self->nom = chaine_dupliquer(nom);
    self->nombre_d_articles = nombre_d_articles;

    return self;
}

void
client_liberer(client self)
{
    memoire_liberer(self->nom);
    memoire_liberer(self);
}

int
client_nombre_d_article(client self)
{
    return self->nombre_d_articles;
}

char *
client_nom(client self)
{
    return self->nom;
}

void
client_decrementer_nombre_d_articles(client self)
{
    if (self->nombre_d_articles > 0)
	self->nombre_d_articles--;
}

void
client_afficher(client self)
{
    if (self == NULL)
	return;
    printf("<\"%s\"(%d)> ", self->nom, self->nombre_d_articles);
}
