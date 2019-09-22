#ifndef TSP_H
#define TSP_H

#include "matrice2d.h"

typedef struct chemin *chemin;

typedef struct {double x, y;} coordonnees;
typedef struct {int s,t;} arete;


void chemin_liberer(chemin tour);
void chemin_afficher (chemin tour);
int chemin_element (chemin tour, int position);
void chemin_copier (chemin src, chemin dst);

double chemin_longueur(chemin tour, matrice2d dist);

void sauver_tsplib(char * nom_fichier, matrice2d);
matrice2d charger_tsplib(char * nom_fichier);
matrice2d charger_villes_L2(char * nom_fichier);
matrice2d charger_villes_Linf(char * nom_fichier);


chemin nearest_neighbor_tsp(matrice2d distances);
chemin mst_tsp(matrice2d distances);
chemin exhaustive_tsp(matrice2d distances);
chemin exhaustive_BB_tsp(matrice2d distances);

arete* calcul_mst(matrice2d distances);


coordonnees* lire_villes(char *nom_fichier, int *nb_villes);

#endif
