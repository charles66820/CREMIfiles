#include "tools.h"
//#include "tsp_brute_force.h"
#include "tsp_prog_dyn.h"
//#include "tsp_heuristic.h"
//#include "tsp_mst.h"

int main(int argc, char *argv[]) {
  const int n = (argc >= 2) ? atoi(argv[1]) : 10;
  unsigned seed = time(NULL) % 1000;
  printf("seed: %u\n", seed);  // pour rejouer la même chose au cas où
  srandom(seed);

  TopChrono(0);  // initialise tous les chronos

  point *V = NULL;        // tableau de points
  V = generatePoints(n);  // n points au hasard
  // height=140*2.2, width=height*1.2; // rayon=height/2.2
  // int k=5; V = generateCircles(n,k); // n points sur k cercles

  int *P = malloc(n * sizeof(int));  // P = la tournée
  P[0] = -1;  // permutation qui ne sera pas dessinée par drawTour()

  init_SDL_OpenGL();     // initialisation avant de dessiner
  drawTour(V, n, NULL);  // dessine seulement les points

#ifdef TSP_BRUTE_FORCE_H
  printf("*** brute-force ***\n");
  running = true;  // force l'exécution
  TopChrono(1);    // départ du chrono 1
  printf("value: %g\n", tsp_brute_force(V, n, P));
  printf("running time: %s\n", TopChrono(1));  // durée
  printf("waiting for a key ... ");
  fflush(stdout);
  update = true;        // force l'affichage
  while (running) {     // affiche le résultat et attend (q pour sortir)
    drawTour(V, n, P);  // dessine la tournée
    handleEvent(true);  // attend un évènement (=true) ou pas
  }
  printf("\n");

  printf("*** brute-force optimisé ***\n");
  running = true;  // force l'exécution
  TopChrono(1);    // départ du chrono 1
  printf("value: %g\n", tsp_brute_force_opt(V, n, P));
  printf("running time: %s\n", TopChrono(1));  // durée
  printf("waiting for a key ... ");
  fflush(stdout);
  update = true;        // force l'affichage
  while (running) {     // affiche le résultat et attend (q pour sortir)
    drawTour(V, n, P);  // dessine la tournée
    handleEvent(true);  // attend un évènement (=true) ou pas
  }
  printf("\n");
#endif

#ifdef TSP_PROG_DYN_H
  printf("*** programmation dynamique ***\n");
  running = true;  // force l'exécution
  TopChrono(1);    // départ du chrono 1
  printf("value: %g\n", tsp_prog_dyn(V, n, P));
  printf("running time: %s\n", TopChrono(1));  // durée
  printf("waiting for a key ... ");
  fflush(stdout);
  update = true;        // force l'affichage
  while (running) {     // affiche le résultat et attend (q pour sortir)
    drawTour(V, n, P);  // dessine la tournée
    handleEvent(true);  // attend un évènement (=true) ou pas
  }
  printf("\n");
#endif

#ifdef TSP_HEURISTIC_H
  printf("*** flip ***\n");
  running = true;  // force l'exécution
  TopChrono(1);    // départ du chrono 1
  printf("value: %g\n", tsp_flip(V, n, P));
  printf("running time: %s\n", TopChrono(1));  // durée
  printf("waiting for a key ... ");
  fflush(stdout);
  while (running) {  // affiche le résultat et attend (q pour sortir)
    update = (first_flip(V, n, P) == 0.0);  // force l'affichage si pas de flip
    drawTour(V, n, P);                      // dessine la tournée
    handleEvent(update);  // attend un évènement (si affichage) ou pas
  }
  printf("\n");

  printf("*** greedy ***\n");
  running = true;  // force l'exécution
  TopChrono(1);    // départ du chrono 1
  printf("value: %g\n", tsp_greedy(V, n, P));
  printf("running time: %s\n", TopChrono(1));  // durée
  printf("waiting for a key ... ");
  fflush(stdout);
  update = true;          // force l'affichage
  while (running) {       // affiche le résultat et attend (q pour sortir)
    drawTour(V, n, P);    // dessine la tournée
    handleEvent(update);  // attend un évènement (si affichage) ou pas
    update = (first_flip(V, n, P) == 0.0);  // force l'affichage si pas de flip
  }
  printf("\n");
#endif

#ifdef TSP_MST_H
  printf("*** mst ***\n");
  running = true;            // force l'exécution
  TopChrono(1);              // départ du chrono 1
  graph T = createGraph(n);  // graphe vide
  printf("value: %g\n", tsp_mst(V, n, P, T));
  printf("running time: %s\n", TopChrono(1));  // durée
  printf("waiting for a key ... ");
  fflush(stdout);
  bool new_redraw = true;
  update = true;  // force l'affichage
  while (running) {
    if (new_redraw) tsp_mst(V, n, P, T);
    drawGraph(V, n, P, T);
    new_redraw =
        handleEvent(update);  // attend un évènement (si affichage) ou pas
    update = (first_flip(V, n, P) == 0.0);  // force l'affichage si pas de flip
  }
  freeGraph(T);
  printf("\n");
#endif

  // Libération de la mémoire
  TopChrono(-1);
  free(V);
  free(P);

  cleaning_SDL_OpenGL();
  return 0;
}
