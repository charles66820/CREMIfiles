#include "tsp.h"
#include <stdio.h>
#include <string.h>

void usage(char *nom) {
  printf("%s NN|MST|EX|BB <nom_fichier>\n", nom);
}

int main(int argc, char* argv[]) {
  if(argc !=3) {
    usage(argv[0]);
    return -1;
  }
  
  matrice2d dist = charger_tsplib(argv[2]);
  printf("chargement OK. Taille du problème : %d  \n", m2d_taille(dist));
  char *algo = argv[1];
  chemin tour;
  if (strcmp(algo,"NN")==0)
    tour = nearest_neighbor_tsp(dist);
  else if (strcmp(algo,"MST")==0)
    tour = mst_tsp(dist);
  else if (strcmp(algo,"EX")==0)
    tour =  exhaustive_tsp(dist);
  else if (strcmp(algo,"BB")==0)
    tour = exhaustive_BB_tsp(dist);
  else {
    usage(argv[0]);
    return -1;
  }
  
  printf("longueur  %lf\n", chemin_longueur(tour, dist));
  chemin_afficher(tour);
  printf("fin\n");
  
  return 0;
}
