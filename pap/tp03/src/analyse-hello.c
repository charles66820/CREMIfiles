#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main() {
#pragma omp parallel
  {
    int me = omp_get_thread_num();

#pragma omp task firstprivate(me)
    printf("Bonjour de %d exécuté par %d\n", me, omp_get_thread_num());

#pragma omp task firstprivate(me)
    printf("Au revoir de %d exécuté par %d\n", me, omp_get_thread_num());
  }
  return 0;
}
