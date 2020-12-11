#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

unsigned long volatile k = 0;
const unsigned long MAX = 100 * 1000;

void *for_in_parallel(void *p) {
  for (unsigned long i = 0; i < MAX; i++) k++;
  return NULL;
}
// 1 une valeur entre 100 * 1000 et 100 * 1000 * nbThread car la meme valeut peut être écrit plusieur fois en même temps
// 2 100000
// 3 il faut attendre que chaque thread et fini. un par un

int main(int argc, char *argv[]) {
  int n = atoi(argv[1]);
  pthread_t tids[n];

  for (uint i = 0; i < n; i++)
    pthread_create(tids + i, NULL, for_in_parallel, NULL);

  for (uint i = 0; i < n; i++) pthread_join(tids[i], NULL);

  printf("%lu\n", k);

  return EXIT_SUCCESS;
}
