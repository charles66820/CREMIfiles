#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int id = 0;

void creer_tache(char *nom, int maman) {
#pragma omp task firstprivate(nom, maman)
  {
    sleep(1);
    printf("%s [fille de %d]\n", nom, maman);
  }
}

int main() {
#pragma omp parallel num_threads(4)
  {
    int me;
#pragma omp atomic capture
    me = id++;
#pragma omp single nowait
    {
      printf("tache %d va créer A et B \n", me);
      creer_tache("A", me);
      creer_tache("B", me);
    }

#pragma omp taskwait
#pragma omp single nowait
    {
      printf("tâche %d va créer C et D \n", me);
      creer_tache("C", me);
      creer_tache("D", me);
    }

#pragma omp taskwait
    printf("tache %d a passé taskwait \n", me);
  }
  return 0;
}
