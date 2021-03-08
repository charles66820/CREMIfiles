#include "tsp_brute_force.h"

#include <float.h>

//
//  TSP - BRUTE-FORCE
//
// -> la structure "point" est définie dans tools.h

double dist(point A, point B) {
  double dx = A.x - B.x;
  double dy = A.y - B.y;
  return sqrt(dx * dx + dy * dy);
}

double value(point *V, int n, int *P) {
  double distSum = 0;
  for (int i = 0; i < n; i++) distSum += dist(V[P[i]], V[P[i + 1] % n]);
  return distSum;
}

double tsp_brute_force(point *V, int n, int *Q) {
  int P[n];
  for (int i = 0; i < n; i++) P[i] = i;  // Create the first permutation
  double bestValue = DBL_MAX;

  // Test all permutation
  do {
    double currentValue = value(V, n, P);
    if (currentValue < bestValue) {
      bestValue = currentValue;
      memcpy(Q, P, n * sizeof(int));  // copy P in Q
    }
    drawTour(V, n, P); // Show
  } while (NextPermutation(P, n));

  return bestValue;
}

int compare(const void *a, const void *b) {
  int int_a = *((int *)a);
  int int_b = *((int *)b);
  return (int_a < int_b) - (int_a > int_b);
}

void MaxPermutation(int *P, int n, int k) {
  // Sort from prefix to end
  qsort(P + k, n - k, sizeof(int), compare);
}

double value_opt(point *V, int n, int *P, double w) {
  double s = 0;
  for (int i = 0; i < n - 1; i++) {
    if (s + dist(V[P[i]], V[P[0]]) >= w) return -(i + 1);
    s += dist(V[P[i]], V[P[i + 1]]);
  }
  return s + dist(V[P[n - 1]], V[P[0]]);
}

double tsp_brute_force_opt(point *V, int n, int *Q) {
  int P[n];
  for (int i = 0; i < n; i++) P[i] = i;  // Create the first permutation
  double bestValue = DBL_MAX;

  // Test all permutation
  do {
    double w = value_opt(V, n, P, bestValue); // currentValue is w
    if (w < 0) {
      MaxPermutation(P, n, abs(w)); // TODO: bug with k ? or w in value_opt
    } else if (w < bestValue) { // bestValue is mim
      bestValue = w;
      memcpy(Q, P, n * sizeof(int));  // copy P in Q
    }
    drawTour(V, n, P); // Show
  } while (NextPermutation(P, n));

  return bestValue;
}
