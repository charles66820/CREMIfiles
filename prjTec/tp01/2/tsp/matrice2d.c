#include "matrice2d.h"
#include "memoire.h"
#include <stdio.h>
#include <assert.h>

struct matrice2d {
  VAL_TYPE **val;
  int taille;
};

matrice2d m2d_creer(int taille) {
  matrice2d self = memoire_allouer(sizeof(struct matrice2d));
  self->taille = taille;
  self->val = memoire_allouer(taille * sizeof(VAL_TYPE*));
  self->val[0]= memoire_allouer(taille*taille*sizeof(VAL_TYPE));
  for(int i=1; i < taille; i++)
    self->val[i]=self->val[0]+i*taille;
  return self;
}

void m2d_afficher(matrice2d self) {
  for (int i = 0; i < self->taille; i++) {
    printf("\n");
    for (int j = 0; j < self->taille; j++) {
      printf("%g ",self->val[i][j]);
    }
  }
}


void m2d_liberer(matrice2d self) { 
  memoire_liberer(self->val[0]);
  memoire_liberer(self->val);
  memoire_liberer(self);
} 

VAL_TYPE m2d_val(matrice2d self, int i, int j) {
  assert( 0<= i && i < self->taille);
  assert( 0<= j && j < self->taille);
  return self->val[i][j];
}

void m2d_ecrire(matrice2d self, int i, int j, VAL_TYPE valeur) {
  assert( 0<= i && i < self->taille);
  assert( 0<= j && j < self->taille);
  self->val[i][j] = valeur;

}
int m2d_taille(matrice2d self) {
  return self->taille;
}
