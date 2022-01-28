#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int f(int i) {
  sleep(1);
  return 2 * i;
}

int g(int i) {
  sleep(1);
  return 2 * i + 1;
}

int main() {
  int x;
  int y;
#pragma omp parallel // edit
#pragma omp single // edit
  {
#pragma omp task // edit
    x = f(2);

#pragma omp task // edit
    y = g(3);
  }
// #pragma omp taskwait // not need because implicit barrier after parallel
  printf("résultat %d\n", x + y);
  return 0;
}
