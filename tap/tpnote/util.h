#include "tools.h"

// Macro pour simplifier les expressions de calcul des distances
#define D(a, b) (dist(V[P[a]], V[P[b]]))

double dist(point A, point B);

double value(point *V, int n, int *P);

void print_tab(int *P, int n);
