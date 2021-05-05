#ifndef TSP_MIN_H
#define TSP_MIN_H

#include <assert.h>

#include "tools.h"

double dist(point A, point B);

double value(point *V, int n, int *P);

void print_P(int *P, int n);

double tsp_cheapest(point *V, int n, int *P);

double first_flip25(point *V, int n, int *P);

double tsp_flip25(point *V, int n, int *P);

#endif /* TSP_MIN.COR_H */
