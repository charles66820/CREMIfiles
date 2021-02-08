#ifndef TSP_PROG_DYN_H
#define TSP_PROG_DYN_H

#include "tools.h"

/* une cellule de la table */
typedef struct {
  double length; // longueur du chemin minimum D[t][S]
  int pred;      // point précédant t dans la solution D[t][S]
} cell;

int DeleteSet(int S, int i);
int ExtractPath(cell **D, int t, int S, int n, int *Q);
double tsp_prog_dyn(point *V, int n, int *Q);

// pour .cor.c
int NextSet(int S, int n) ;
double tsp_prog_dyn2(point *V, int n, int *Q);
double tsp_prog_dyn3(point *V, int n, int *Q);
double tsp_prog_dyn4(point *V, int n, int *Q);

#endif /* TSP_PROG_DYN_H */
