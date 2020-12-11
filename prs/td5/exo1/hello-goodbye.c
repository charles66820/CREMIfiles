#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

pthread_barrier_t barrier;

void *HelloGoodbye(void *p) {
  printf("%d: bonjour\n", (int)(size_t)p);
  pthread_barrier_wait(&barrier);
  printf("%d: a bientot\n", (int)(size_t)p);

  return NULL;
}

int main(int argc, char *argv[]) {
  int nbThread = 1;

  if (argc > 1) nbThread = atoi(argv[1]);

  pthread_t tid[nbThread];

  // Define barrier
  pthread_barrier_init(&barrier, NULL, nbThread);

  for (uint i = 0; i < nbThread; i++) {
    pthread_create(&tid[i], NULL, HelloGoodbye, (void *)(size_t)i);
  }


  for (uint i = 0; i < nbThread; i++)
    pthread_join(tid[i], NULL);

  return EXIT_SUCCESS;
}
