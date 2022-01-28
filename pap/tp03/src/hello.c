#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main() {
#pragma omp parallel
  {
#pragma omp task
    printf("Bonjour de la part de %d\n", omp_get_thread_num());

#pragma omp task
    printf("Au revoir de la part de %d\n", omp_get_thread_num());
  }
  return 0;
}
