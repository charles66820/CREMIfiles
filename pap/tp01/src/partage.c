#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
  int k = 0;

#pragma omp parallel
  {
    int i;
    for (i = 0; i < 100000; i++)
#pragma omp critical
      k++;
  }

  printf("nbthreads x 100000 = %d\n ", k);
  return 0;
}
