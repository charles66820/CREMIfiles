#ifndef TSP_BRUTE_FORCE_H
#define TSP_BRUTE_FORCE_H

#include "tools.h"

double dist(point A, point B);

double value(point *V, int n, int *P);

double tsp_brute_force(point *V, int n, int *Q);

double value_opt(point *V, int n, int *P, double wmin);

void MaxPermutation(int *P, int n, int k);

double tsp_brute_force_opt(point *V, int n, int *Q);

double value_opt2(point *V, int n, int *P, double wmin);

double tsp_brute_force_opt2(point *V, int n, int *Q);

#endif /* TSP_BRUTE_FORCE_H */
