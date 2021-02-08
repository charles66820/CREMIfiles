#include <float.h>
#include "tsp_brute_force.h"

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
  } while (NextPermutation(P, n));

  return bestValue;
}

void MaxPermutation(int *P, int n, int k) {
  printf("Permutation : ");
  for (int i = 0; i < n; i++) printf("%d ", P[i]);
  printf("\n");

  int Per[n];
  // from prefix to end
  for (int i = k; i < n; i++) Per[i] = P[i];
  // TODO: idk

  printf("Permutation : ");
  for (int i = 0; i < n; i++) printf("%d ", Per[i]);
  printf("\n");

  return;
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
  for (int i = 0; i < n; i++) P[i] = i;
  MaxPermutation(P, n, 4);
  // TODO: idk
  ;
  ;
  ;
  return 0;
}
