#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
  int i;

  for (i = 0; i < 40; i++) printf("%d traite %i\n", omp_get_thread_num(), i);

  return 0;
}
