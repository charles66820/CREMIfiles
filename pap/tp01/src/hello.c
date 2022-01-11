#include <omp.h>
#include <stdio.h>

int main() {
#pragma omp parallel
  {
    printf("Bonjour !(%d)\n", omp_get_thread_num());
#pragma omp barrier
    printf("Au revoir !(%d)\n", omp_get_thread_num());
  }

  return 0;
}
