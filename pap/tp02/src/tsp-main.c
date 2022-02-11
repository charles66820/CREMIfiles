#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define MAX_NBVILLES 15000  // 22

typedef int DTab_t[MAX_NBVILLES][MAX_NBVILLES];
typedef int chemin_t[MAX_NBVILLES];

/* macro de mesure de temps, retourne une valeur en �secondes */
#define TIME_DIFF(t1, t2) \
  ((t2.tv_sec - t1.tv_sec) * 1000000 + (t2.tv_usec - t1.tv_usec))

/* dernier minimum trouv� */
int minimum = INT_MAX;

/* tableau des distances */
DTab_t distance;

/* nombre de villes */
int nbVilles;

/* profondeur du parallélisme */
int grain;

#define MAXX 100
#define MAXY 100
typedef struct {
  int x, y;
} coor_t;

typedef coor_t coortab_t[MAX_NBVILLES];

void initialisation(int Argc, char *Argv[]) {
  if (Argc < 4 || Argc > 5) {
    fprintf(stderr, "Usage: %s  <nbVilles> <seed> [grain] <kernel>\n", Argv[0]);
    exit(1);
  }

  grain = (Argc == 5) ? atoi(Argv[3]) : 0;

  /* initialisation du tableau des distances */
  /* on positionne les villes aléatoirement sur une carte MAXX x MAXY  */

  coortab_t lesVilles;

  int i, j;
  int dx, dy;

  nbVilles = atoi(Argv[1]);
  if (nbVilles > MAX_NBVILLES) {
    fprintf(stderr, "trop de villes, augmentez MAX_NBVILLES\n");
    exit(1);
  }

  srand(atoi(Argv[2]));

  for (i = 0; i < nbVilles; i++) {
    lesVilles[i].x = rand() % MAXX;
    lesVilles[i].y = rand() % MAXY;
  }

  for (i = 0; i < nbVilles; i++)
    for (j = 0; j < nbVilles; j++) {
      dx = lesVilles[i].x - lesVilles[j].x;
      dy = lesVilles[i].y - lesVilles[j].y;
      distance[i][j] = (int)sqrt((double)((dx * dx) + (dy * dy)));
    }
}

/* résolution du problème du voyageur de commerce */

inline int present(int ville, int mask) { return mask & (1 << ville); }

void verifier_minimum(int lg, chemin_t chemin) {
  if (lg + distance[0][chemin[nbVilles - 1]] < minimum) {
    minimum = lg + distance[0][chemin[nbVilles - 1]];
    printf("%3d :", minimum);
    for (int i = 0; i < nbVilles; i++) printf("%2d ", chemin[i]);
    printf("\n");
  }
}

void tsp_seq(int etape, int lg, chemin_t chemin, int mask) {
  if (lg + distance[0][chemin[etape - 1]] >= minimum) return;

  int ici, dist;

  if (etape == nbVilles)
    verifier_minimum(lg, chemin);
  else {
    ici = chemin[etape - 1];

    for (int i = 1; i < nbVilles; i++) {
      if (!present(i, mask)) {
        chemin[etape] = i;
        dist = distance[ici][i];
        tsp_seq(etape + 1, lg + dist, chemin, mask | (1 << i));
      }
    }
  }
}

void tsp_ompfor(int etape, int lg, chemin_t chemin, int mask) {
  if (lg + distance[0][chemin[etape - 1]] >= minimum) return;

  if (etape > grain)
    tsp_seq(etape, lg, chemin, mask);
  else {
#pragma omp parallel num_threads(nbVilles - etape)
    {
      chemin_t monChemin;
      memcpy(monChemin, chemin, etape * sizeof(chemin[0]));
      int ici = chemin[etape];
#pragma omp for schedule(dynamic)
      for (size_t i = 1; i < nbVilles; i++) {
        // here ?
        // if (lg + distance[ici][0] < minimum) {
        //   minimum = distance[ici][0] + lg;
        //   printf("%3d :", minimum);
        //   for (int i = 0; i < nbVilles; i++) printf("%2d ", chemin[i]);
        //   printf("\n");
        // }

        if (!present(i, mask)) {
          monChemin[etape + 1] = i;
          tsp_ompfor(etape + 1, lg + distance[ici][i], monChemin,
                     mask | (1 << i));
        }
      }
    }
  }
}

/*
void tsp_ompfor(int etape, int lg, chemin_t chemin, int mask) {
  if (etape == nbVilles)
    verifier_minimum(lg, chemin);
  else if (etape > grain)
    tsp_seq(etape, lg, chemin, mask);
  else {
    int ici = chemin[etape - 1];
#pragma omp parallel for schedule(dynamic)
    for (int i = 1; i < nbVilles; i++) {
      if (!present(i, mask)) {
        chemin_t mon_chemin;
        memcpy(mon_chemin, chemin, etape * sizeof(*chemin));
        mon_chemin[etape] = i;
        int dist = distance[ici][i];
        tsp_ompfor(etape + 1, lg + dist, mon_chemin, mask | (1 << i));
      }
    }
  }
}
*/

void tsp_task(int etape, int lg, chemin_t chemin, int mask) {
  if (etape == nbVilles)
    verifier_minimum(lg, chemin);
  else if (etape > grain)
    tsp_seq(etape, lg, chemin, mask);
  else {
    int ici = chemin[etape - 1];
#pragma omp parallel for schedule(dynamic)
    for (int i = 1; i < nbVilles; i++) {
      if (!present(i, mask))
#pragma omp task shared(chemin) // edit
      {
        chemin_t mon_chemin;
        memcpy(mon_chemin, chemin, etape * sizeof(*chemin));
        mon_chemin[etape] = i;
        int dist = distance[ici][i];
        tsp_task(etape + 1, lg + dist, mon_chemin, mask | (1 << i));
      }
    }
#pragma omp taskwait // edit
  }
}

// malloc version
void tsp_task2(int etape, int lg, chemin_t chemin, int mask) {
  if (etape == nbVilles)
    verifier_minimum(lg, chemin);
  else if (etape > grain)
    tsp_seq(etape, lg, chemin, mask);
  else {
    int ici = chemin[etape - 1];
#pragma omp parallel for schedule(dynamic)
    for (int i = 1; i < nbVilles; i++) {
      if (!present(i, mask))
      {
        chemin_t* mon_chemin = malloc(nbVilles * sizeof(*chemin)) ;
        memcpy(mon_chemin, chemin, etape * sizeof(*chemin));
        #pragma omp task firstprivate(mon_chemin)
        {
          *mon_chemin[etape] = i;
          int dist = distance[ici][i];
          tsp_task2(etape + 1, lg + dist, *mon_chemin, mask | (1 << i));
        }
        free(mon_chemin);
      }
    }
  }
}

void tsp_task3(int etape, int lg, chemin_t chemin, int mask) {
  if (etape == nbVilles)
    verifier_minimum(lg, chemin);
  else if (etape > grain)
    tsp_seq(etape, lg, chemin, mask);
  else {
    int ici = chemin[etape - 1];
      chemin_t mon_chemin;
      memcpy(mon_chemin, chemin, etape * sizeof(*chemin));
#pragma omp parallel for schedule(dynamic)
    for (int i = 1; i < nbVilles; i++) {
      if (!present(i, mask))
#pragma omp task firstprivate(mon_chemin)
      {
        mon_chemin[etape] = i;
        int dist = distance[ici][i];
        tsp_task3(etape + 1, lg + dist, mon_chemin, mask | (1 << i));
      }
    }
  }
}

// collaps.c

void tsp_ompcol2() {
  int i, j;
#pragma omp parallel for collapse(2) schedule(runtime)
  for (i = 1; i < nbVilles; i++)
    for (j = 1; j < nbVilles; j++)
      if (i != j) {
        chemin_t chemin;
        chemin[0] = 0;
        chemin[1] = i;
        chemin[2] = j;
        int dist = distance[0][i] + distance[i][j];
        tsp_seq(3, dist, chemin, 1 | (1 << i) | (1 << j));
      }
}

void tsp_ompcol3() {
  int i, j, k;
#pragma omp parallel for collapse(3) schedule(runtime)
  for (i = 1; i < nbVilles; i++)
    for (j = 1; j < nbVilles; j++)
      for (k = 1; k < nbVilles; k++)
        if (i != j && i != k && j != k) {
          chemin_t chemin;
          chemin[0] = 0;
          chemin[1] = i;
          chemin[2] = j;
          chemin[3] = k;
          int dist = distance[0][i] + distance[i][j] + distance[j][k];
          tsp_seq(4, dist, chemin, 1 | (1 << i) | (1 << j) | (1 << k));
        }
}

void tsp_ompcol4() {
  int i, j, k;
#pragma omp parallel for collapse(4) schedule(runtime)
  for (i = 1; i < nbVilles; i++)
    for (j = 1; j < nbVilles; j++)
      for (k = 1; k < nbVilles; k++)
        for (int l = 1; l < nbVilles; l++)
          if (i != j && i != k && j != k && i != l && j != l && k != l) {
            chemin_t chemin;
            chemin[0] = 0;
            chemin[1] = i;
            chemin[2] = j;
            chemin[3] = k;
            chemin[4] = l;
            int dist = distance[0][i] + distance[i][j] + distance[j][k] +
                       distance[k][l];
            tsp_seq(5, dist, chemin,
                    1 | (1 << i) | (1 << j) | (1 << k) | (1 << l));
          }
}

int main(int argc, char **argv) {
  unsigned long temps;
  struct timeval t1, t2;
  chemin_t chemin;

  initialisation(argc, argv);

  printf("nbVilles = %3d - grain %d \n", nbVilles, grain);

  // omp_set_max_active_levels(grain);
  omp_set_nested(1);

  gettimeofday(&t1, NULL);

  chemin[0] = 0;

  if (!strcmp(argv[argc - 1], "seq"))
    tsp_seq(1, 0, chemin, 1);
  else if (!strcmp(argv[argc - 1], "ompfor"))
    tsp_ompfor(1, 0, chemin, 1);
  else if (!strcmp(argv[argc - 1], "task1"))
    tsp_task(1, 0, chemin, 1);
  else if (!strcmp(argv[argc - 1], "task2"))
    tsp_task2(1, 0, chemin, 1);
  else if (!strcmp(argv[argc - 1], "task3"))
    tsp_task3(1, 0, chemin, 1);
  else if (!strcmp(argv[argc - 1], "ompcol2"))
    tsp_ompcol2(1, 0, chemin, 1);
  else if (!strcmp(argv[argc - 1], "ompcol3"))
    tsp_ompcol3(1, 0, chemin, 1);
  else if (!strcmp(argv[argc - 1], "ompcol4"))
    tsp_ompcol4(1, 0, chemin, 1);
  else {
    printf("kernel inconnu\n");
    exit(1);
  }

  gettimeofday(&t2, NULL);

  temps = TIME_DIFF(t1, t2);
  fprintf(stderr, "%ld.%03ld\n", temps / 1000, temps % 1000);

  return 0;
}
