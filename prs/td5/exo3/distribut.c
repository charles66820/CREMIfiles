#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#include "distributor.h"

// La consigne est trop flou

const unsigned long MAX = 100 * 1000;

pthread_barrier_t barrier;

void *for_in_parallel(void *p) {
  for (unsigned long i = 0; i < MAX; i++) {
    distributor_next();
    pthread_barrier_wait(&barrier);
  }
  return NULL;
}

int main(int argc, char *argv[]) {
  int n = atoi(argv[1]);
  pthread_t tids[n];

  // Define barrier
  pthread_barrier_init(&barrier, NULL, n);

  for (int i = 0; i < n; i++)
    pthread_create(tids + i, NULL, for_in_parallel, NULL);

  for (int i = 0; i < n; i++) pthread_join(tids[i], NULL);

  printf("%d\n", distributor_value());

  return EXIT_SUCCESS;
}
