#ifndef MATRICE2D_H
#define MATRICE2D_H

typedef struct matrice2d *matrice2d;

#define VAL_TYPE double

matrice2d m2d_creer(int taille);

void m2d_afficher(matrice2d self);

void m2d_liberer(matrice2d self);

VAL_TYPE m2d_val(matrice2d self, int i, int j);

void m2d_ecrire(matrice2d self, int i, int j, VAL_TYPE valeur);

int m2d_taille(matrice2d self);


#endif
