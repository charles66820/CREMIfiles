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
  int x = f(2);
  int y = g(3);
  printf("résultat %d\n", x + y);
  return 0;
}
