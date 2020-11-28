#define XOPEN_SOURCE 600

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define NSIGNORT 32

// 1 done
// 2 we have many signal ignored and somtime the process close before the
// handler as set 3 a: because the handler need to be set before send signal
// (before create the child sender)
//   b: because mask signal permite to handle signals later

int sender(int fatherPid, int argc, char *argv[]) {
  int k = atoi(argv[1]);

  sleep(1);

  for (int i = 0; i < k; i++)
    for (int j = 2; j < argc; j++) {
      kill(fatherPid, atoi(argv[j]));
    }

  // kill(fatherPid, SIGKILL);
  return EXIT_SUCCESS;
}

int sigtab[NSIGNORT];

void sigHandler(int sig) {
  sigtab[sig - 1]++;
  printf("Signal n°%d %s\n", sigtab[sig - 1], strsignal(sig));
}

int main(int argc, char *argv[]) {
  // Fill sigtab with 0
  for (uint i = 0; i < NSIGNORT; i++) sigtab[i] = 0;

  // Define handler
  struct sigaction sa;
  sa.sa_flags = SA_NOCLDSTOP;
  sigfillset(&sa.sa_mask);
  sa.sa_handler = sigHandler;
  sigprocmask(SIG_BLOCK, &sa.sa_mask, NULL); // block all signals



  pid_t pid = fork();
  if (!pid)
    sender(getppid(), argc, argv);
  else {
    printf("receiver : %d\n", getpid());

    // Set handler
    for (int sig = 0; sig < NSIGNORT; sig++) {
      sigaction(sig, &sa, NULL);
    }
    sigprocmask(SIG_UNBLOCK, &sa.sa_mask, NULL); // unblock all signals

    while (1) pause();

    return EXIT_SUCCESS;
  }
}
