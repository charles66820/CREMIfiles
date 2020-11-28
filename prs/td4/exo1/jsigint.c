#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

struct sigaction sa, oldSa;

void sigIntHandler(int sig) {
  printf("ctrl-c %d %s\n", sig, strsignal(sig));
  // Disable SIGINT handler
  sigaction(SIGINT, &oldSa, NULL);
}

int main(int argc, char* argv[]) {
  sa.sa_flags = 0;
  sigemptyset(&sa.sa_mask); // empty the mask
  sa.sa_handler = sigIntHandler;
  sigaction(SIGINT, &sa, &oldSa);

/*
  sigaddset(&sa.sa_mask, SIGINT); // add to mask
  sigprocmask(SIG_BLOCK, &sa.sa_mask, &oldSa.sa_mask); // apply mask
  sleep(2);
  sigprocmask(SIG_UNBLOCK, &sa.sa_mask, NULL); // apply mask
*/
  while (1);

  return EXIT_SUCCESS;
}



