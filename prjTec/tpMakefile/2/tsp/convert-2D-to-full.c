#include "tsp.h"
#include "stdio.h"

int main(int argc, char* argv[]) {
  if(argc !=3) {
    printf("%s <nom_fichier_entree> <nom_fichier_sortie>\n", argv[0]);
    return -1;
  }
    
  matrice2d dist = charger_villes_L2(argv[1]);
  printf("chargement OK \n");
  sauver_tsplib(argv[2], dist);
  printf("conversion OK \n");
  
  return 0;
}
