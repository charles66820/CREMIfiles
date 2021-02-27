#ifndef TSP_HEURISTIC_H
#define TSP_HEURISTIC_H

#include "tools.h"

void reverse(int *T, int p, int q);
double first_flip(point *V, int n, int *P);
double tsp_flip(point *V, int n, int *P);
double tsp_greedy(point *V, int n, int *P);

#endif /* TSP_HEURISTIC_H */
