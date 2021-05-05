//
// TAP - TP NOTE 2021
// HEURISTIQUE CHEAPEST INSERTION
//

#include "tp.h"

// Constantes à utiliser dans la fonction tsp_cheapest pour l'animation
#define ANIMATION 1        // Animation on/off pour algo cheapest.
#define ANIMATION_DELAY 2  // Pause, en secondes

////////////////////////
// FONCTIONS FOURNIES //
////////////////////////

// Distance Euclidienne
double dist(point A, point B);

// Valeur d'une tournée P de taille n.
double value(point *V, int n, int *P);

// Affichage d'un tableau d'entiers P de n cases.
void print_tab(int *P, int n);

//////////////////////////////////////////
// FONCTIONS A COMPLETER A PARTIR D'ICI //
//////////////////////////////////////////

// Exercice 1
void nearest_points(point *V, int n, int *first_ptr, int *second_ptr) {
  if (n < 2) return;  // Error

  int first = 0;
  int second = 1;
  int dmin = dist(V[first], V[second]);

  for (size_t i = 1; i < n; i++)
    for (size_t j = i + 1; j < n; j++) {
      int d = dist(V[i], V[j]);
      if (d < dmin) {
        dmin = d;
        first = i;
        second = j;
      }
    }

  if (first == second) return;
  *first_ptr = first;
  *second_ptr = second;
}

// Exercice 2
void init_tour(point *V, int n, int *P) {
  // Init P
  for (size_t i = 0; i < n; i++) P[i] = i;

  int first = 0;
  int second = 1;
  nearest_points(V, n, &first, &second);

  int tmp;
  SWAP(P[0], P[first], tmp);
  SWAP(P[1], P[second], tmp);
}

// Exercice 5
double score(point *V, int m, int *P, int i, int *pred_ptr) {
  if (i < m) return 0.0;  // Error

  int score = dist(V[P[0]], V[P[i]]) + dist(V[P[i]], V[P[(1) % m]]);
  int kMin = 0;

  for (size_t k = 1; k < m; k++) {
    int newScore = dist(V[P[k]], V[P[i]]) + dist(V[P[i]], V[P[(k + 1) % m]]);
    if (newScore < score) {
      score = newScore;
      kMin = k;
    }
  }

  *pred_ptr = kMin;

  return score;
}

// Exercice 6
int new_point(point *V, int n, int m, int *P, int *pred_ptr) {
  if (m >= n) exit(EXIT_FAILURE);  // Error

  int kPred = m;                          // By default k is equal to m
  int minScore = score(V, m, P, m, &kPred);  // begin with i is equal to m
  int newPointI = m;
  for (size_t i = m + 1; i < n; i++) {
    int newKPred = i;
    int newScore = score(V, m, P, i, &newKPred);
    if (newScore < minScore) {
      minScore = newScore;
      kPred = newKPred;
      newPointI = i;
    }
  }

  *pred_ptr = kPred;

  return newPointI;
}

// Exercice 7
void rotate_right(int *P, int p, int q) {
  if (p >= q) return; // Error

  for (size_t i = q; i > p; i--) {
    int tmp;
    SWAP(P[i], P[i - 1], tmp);
  }
}

// Exercice 8
double tsp_cheapest(point *V, int n, int *P) {
  init_tour(V, n, P);

  int length = dist(V[P[0]], V[P[1]]);

  for (size_t m = 2; m < n; m++) {
    int kPred = m;
    int pNew = new_point(V, n, m, P, &kPred);
    rotate_right(P, kPred, pNew);
    if (running && ANIMATION) {
      drawPartialTour(V, n, P, m);
      sleep(ANIMATION_DELAY);
    }
    length += dist(V[P[m - 1]], V[P[m]]);
  }

  return length;
}

// Exercice 10
double gain(point *V, int n, int *P, int i, int j) {
  double initLength = 0.0;
  for (size_t o = 0; o < n - 1; o++) initLength += dist(V[P[o]], V[P[o + 1]]);

  int tmpP[n];
  for (size_t o = 0; o < n; o++) tmpP[o] = P[o];

  rotate_right(tmpP, i, j + 1);
  double moveLength = 0.0;
  for (size_t o = 0; o < n - 1; o++) moveLength += dist(V[tmpP[o]], V[tmpP[o + 1]]);
  return initLength - moveLength;
}

// Exercice 11
double first_flip25(point *V, int n, int *P) {
  double variation = 0.0;
  for (size_t i = 0; i < n - 1; i++)
    for (size_t j = i + 1; j < n - 1; j++) {
      variation = gain(V, n, P, i, j);
      if (variation > 0.0)
        rotate_right(P, i, j + 1);
    }

  return variation;
}

// Exercice 12
double tsp_flip25(point *V, int n, int *P) {
  // Init P
  for (size_t i = 0; i < n; i++) P[i] = i;

  double variation = 0;
  double length = 0;

  while (true) { //Infinit loop
    variation = first_flip25(V, n, P);
    if (variation <= 0) return length;
    /*if (length = 0)
      length += variation;
    else if (variation > 0)
      length -= variation;
    else return length;*/
  }

  return variation;
}
