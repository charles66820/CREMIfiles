#include <stdlib.h>
#include <stdio.h>
#include <signal.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

#include "memoire.h"
#include "chaine.h"
#include "file.h"

#include "client.h"

#define NOMBRE_MAXIMUM_D_ARTICLES 30

static int temps = 0;
static int frequence_d_arrivee;
static int nombre_de_clients = 0;

static int terminer = 0;

static int nombre_de_caisses;
static file *caisses;

static int
L_nombre_au_hasard(int max)
{
    int n = rand();
    float x = (float) n / (float) RAND_MAX;
    int r =  floor(x*(max+1));

    if (r > max)
    {
	r--;
	fprintf(stderr, "**\n");
    }
    return r;
}

static char
L_voyelle_au_hasard(void)
{
    static const char voyelle[] = "aeiouy";

    return voyelle[L_nombre_au_hasard(sizeof(voyelle)-2)];
}


static char
L_consonne_au_hasard(void)
{
    static const char consonne[] = "bcdfghijklmnpqrstvwxz";

    return consonne[L_nombre_au_hasard(sizeof(consonne)-2)];
}


static char *
L_nom_au_hasard(void)
{
    static char nom[9];
    int nombre_de_lettres = L_nombre_au_hasard(6) + 2;
    int i = 0;

    for (;;)
    {
	nom[i++] = L_voyelle_au_hasard();
	if (i >= nombre_de_lettres)
	    break;
	nom[i++] = L_consonne_au_hasard();
	if (i >= nombre_de_lettres)
	    break;
    }
    nom[i] = '\0';
    return nom;
}

static char *
L_lire_chaine(char *message)
{
    char chaine[256];

    printf("%s", message);
    fgets(chaine, sizeof(chaine), stdin);
    chaine[strlen(chaine) - 1] = '\0';
    return chaine_dupliquer(chaine);
}

static void
L_afficher_l_etat_des_caisses(void)
{
    int i;

    printf("\033[H\033[2J Temps : [%05d]  -- Nombre de clients : %3d -- "
	   "Fréquence d'arrivée : %3d\n\n", temps,
	   nombre_de_clients, frequence_d_arrivee);

    for (i = 0; i < nombre_de_caisses; i++)
    {
	file f = caisses[i];

	printf("- caisse %2d : ", i);
	if (! file_vide(f))
	    file_iterer(f, client_afficher);
	printf("\n");
    }
    printf("\n------------------------\n\n");
    sleep(1);
}

static client
L_faire_arriver_un_nouveau_client()
{
    char *nom;
    int nombre_d_articles;
    client c;

    nom = L_nom_au_hasard();
    nombre_d_articles = L_nombre_au_hasard(NOMBRE_MAXIMUM_D_ARTICLES);
    c = client_creer(chaine_dupliquer(nom), nombre_d_articles);
    nombre_de_clients++;

    return c;
}

static int
L_choisir_une_caisse(client c)
{
    return L_nombre_au_hasard(nombre_de_caisses - 1);
}

static void
L_traiter_caisse(int numero_de_caisse)
{
    file f = caisses[numero_de_caisse];
    client c;

    if (file_vide(f))
	return;

    c = file_tete(f);
    if (client_nombre_d_article(c) == 0)
    {
	printf("\n --> caisse %d : %s paie et s'en va [ENVOI]",
	       numero_de_caisse, client_nom(c));
	getchar();
	client_liberer(c);
	nombre_de_clients--;
	file_defiler(f);
    }
    else
	client_decrementer_nombre_d_articles(c);
}

static void
L_faire_avancer_les_caisses(void)
{
    int i;

    for (i = 0; i < nombre_de_caisses; i++)
	L_traiter_caisse(i);
}

static void
simulation(void)
{
    while (! terminer)
    {
	temps++;
	
	L_faire_avancer_les_caisses();
	L_afficher_l_etat_des_caisses();
	if (temps % frequence_d_arrivee == 0)
	{
	    client c = L_faire_arriver_un_nouveau_client();
	    int i = L_choisir_une_caisse(c);
	    file f = caisses[i];

	    file_enfiler(f, c);
	    printf("\n --> %s fait la queue à la caisse %d avec %d article(s)"
		   " [ENVOI]", client_nom(c), i, client_nombre_d_article(c));
	    getchar();
	}
    }
}

static void
initialiser_les_caisses(int n)
{
    int i;

    nombre_de_caisses = n;
    caisses = memoire_allouer(nombre_de_caisses*sizeof(file *));
    for (i = 0; i < nombre_de_caisses; i++)
	caisses[i] = file_creer();
}

static void
usage(char *s)
{
    fprintf(stderr, "Usage: %s nombre-de-caisses frequence-d-arrivee-client\n",
	    s);
    exit(EXIT_FAILURE);
}

void
traiter_interruption(int signal)
{
    char *s = L_lire_chaine("\nNouvelle frequence d'arrivée : ");

    frequence_d_arrivee = strtol(s, NULL, 0);
    memoire_liberer(s);
}

int
main(int argc, char *argv[])
{
    if (argc != 3)
	usage(argv[0]);

    initialiser_les_caisses(strtol(argv[1], NULL, 0));
    frequence_d_arrivee = strtol(argv[2], NULL, 0);

    signal(SIGQUIT, traiter_interruption);
    simulation();

    return EXIT_SUCCESS;
}
