#include <setjmp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include "try.h"
#include <unistd.h>

jmp_buf cur;

/*void f(void *p) {
  int x = NULL;
  *x = 4;
}*/

void sigHandler(int sig) {
  siglongjmp(cur, 2);
}

int tryIt(void(*f)(void*), void *p, int sig) {
  // Define signal handler
  struct sigaction sa;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = 0;
  sa.sa_handler = sigHandler;
  sigaction(sig, &sa, NULL);

  jmp_buf buf;
  jmp_buf oldBuf;
  volatile int i;
  if (!(i = sigsetjmp(buf, 1))) {
    oldBuf[0] = cur[0]; // Save old buffer
    cur[0] = buf[0]; // Set current buffer
    f(p);
    cur[0] = oldBuf[0]; // Restore old buffer
  }
  return i;
}

int tryBefore(void (*f)(void *), void *p, int delay) {
  alarm(delay);
  return tryIt(f, p, SIGALRM) == 2? 1 : 0;
}

/* int main(int argc, char *argv[]) {
  int r = tryIt(f, NULL, SIGSEGV);
  if (r == 0)
    printf("L'exécution de f s'est déroulée sans problème\n");
  else
    printf("L'exécution de f a échoué\n");

  return EXIT_SUCCESS;
}*/
