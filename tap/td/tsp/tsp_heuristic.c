#include "tools.h"
#include "tsp_brute_force.h"

//
//  TSP - HEURISTIQUES
//

void reverse(int *T, int p, int q) {
  // Renverse la partie T[p]...T[q] du tableau T avec p<q si
  // T={0,1,2,3,4,5,6} et p=2 et q=5, alors le nouveau tableau T sera
  // {0,1, 5,4,3,2, 6}.
  int tmp;
  for (int i = 0; i <= (q - p) / 2; i++) {
    tmp = T[p + i];
    T[p + i] = T[q - i];
    T[q - i] = tmp;
  }
}

double first_flip(point *V, int n, int *P) {
  // Renvoie le gain>0 du premier flip réalisable, tout en réalisant
  // le flip, et 0 s'il n'y en a pas.
  for (int i = 0; i < n - 2; i++)
    for (int j = i + 2; j < n; j++) {
      point v_i = V[P[i]];
      point v_i1 = V[P[i + 1]];
      point v_j = V[P[j]];
      point v_j1 = V[P[j + 1] % n];
      double dist_actuel = dist(v_i, v_i1) + dist(v_j, v_j1);
      double dist_si_flip = dist(v_i, v_j) + dist(v_i1, v_j1);
      if (dist_actuel > dist_si_flip) {
        reverse(P, i + 1, j);
        return dist_actuel - dist_si_flip;
      }
    }
  return 0.0;
}

double tsp_flip(point *V, int n, int *P) {
  // La fonction doit renvoyer la valeur de la tournée obtenue. Pensez
  // à initialiser P, par exemple à P[i]=i. Pensez aussi faire
  // drawTour() pour visualiser chaque flip.
  for (int i = 0; i < n; i++) {
    P[i] = i;
  }

  while (first_flip(V, n, P) > 0.0) {
    drawTour(V, n, P);
  }
  return value(V, n, P);
}

double tsp_greedy(point *V, int n, int *P) {
  // La fonction doit renvoyer la valeur de la tournée obtenue. Pensez
  // à initialiser P, par exemple à P[i]=i.
  ;
  ;
  ;
  return 0.0;
}
