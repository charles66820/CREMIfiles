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
  bool freePoint[n];
  for (int i = 0; i < n; i++) {
    freePoint[i] = true;
  }
  freePoint[0] = false;

  P[0] = 0;
  for (int i = 0; i < n; i++) {
    int closerFreePoint = 0;
    double min_dist = DBL_MAX;
    for (int j = 0; j < n; j++)
      if (freePoint[j] && j != i) {
        double new_dist = dist(V[i], V[j]);
        if (new_dist < min_dist) {
          min_dist = new_dist;
          closerFreePoint = j;
        }
      }
    if (i != n - 1) P[i + 1] = closerFreePoint;
    freePoint[closerFreePoint] = false;
    drawTour(V, n, P);
  }

  return value(V, n, P);
}

double tsp_greedy(point *V, int n, int *P) {

  // La fonction doit renvoyer la valeur de la tournée obtenue. Pensez

  // à initialiser P, par exemple à P[i]=i.  

  P[0] = 0; // On part de 0

  bool Vb[n];

  for (size_t i = 0; i < n; i++)

  {

    P[i] = i;

    Vb[i] = false;

  }

  Vb[0] = true;

  int nbr = 1;

  double out = 0;

  

  while(nbr < n){

    double distMin = DBL_MAX;

    int iMin = -1;

    

    for(size_t i = 0; i < n; i++){

      double d = dist(V[P[i]], V[P[nbr-1]]);

      if(!Vb[i] && d < distMin && d != 0){

        distMin = d;

        iMin = i;

      }

    }

    if(iMin != -1){

      out += distMin;

    }

    Vb[iMin] = true;

    P[nbr] = iMin;

    nbr++;

  }

  return out;

}
