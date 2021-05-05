#include "util.h"

// Distance Euclidienne
double dist(point A, point B) {
  const double x = A.x - B.x;
  const double y = A.y - B.y;
  return sqrt(x * x + y * y);
}

// Valeur de la tournée
double value(point *V, int n, int *P) {
  double w = 0;
  for (int i = 0; i < n - 1; i++) w += D(i, i + 1);
  return w + D(n - 1, 0);
}

// Affiche le tableau d'entiers P
void print_tab(int *P, int n) {
  for (int i = 0; i < n; ++i) {
    fprintf(stderr, "%d ", P[i]);
  }
  fprintf(stderr, "\n");
}
