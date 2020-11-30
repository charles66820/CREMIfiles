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

int sigtab[NSIGNORT];
pid_t childPid;

void sigUsr1Handler(int sig) {
  printf("Acquittent receive %s\n", strsignal(sig));
}

int sender(int fatherPid, int argc, char *argv[]) {
  // Define handler
  struct sigaction csa;
  sigemptyset(&csa.sa_mask);
  csa.sa_flags = 0;
  csa.sa_handler = sigUsr1Handler;
  sigaction(SIGUSR1, &csa, NULL);

  int k = atoi(argv[1]);

  sleep(1);

  for (int i = 0; i < k; i++)
    for (int j = 2; j < argc; j++) {
      kill(fatherPid, atoi(argv[j]));
      sigsuspend(&csa.sa_mask); // Wait sig
    }

  kill(fatherPid, SIGKILL);
  return EXIT_SUCCESS;
}

void sigHandler(int sig) {
  sigtab[sig]++;
  printf("Signal n°%d %s\n", sigtab[sig], strsignal(sig));
  kill(childPid, SIGUSR1); // Sand ATK can be in while
}

int father(int childPid) {
  printf("receiver : %d\n", getpid());

  // Define handler
  struct sigaction sa;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = 0;
  sa.sa_handler = sigHandler;

  // Set handler
  for (int sig = 0; sig < NSIGNORT; sig++) {
    sigtab[sig] = 0;
    sigaction(sig, &sa, NULL);
  }

  while (1) sigsuspend(&sa.sa_mask); // Wait sig
  return EXIT_SUCCESS;
}

int main(int argc, char *argv[]) {
  if (argc < 4) {
    printf("Usage: %s, <nb signals>, <sig1>, <sig2> ...\n", argv[0]);
    return EXIT_FAILURE;
  }

  // block all signals
  sigset_t full;
  sigfillset(&full);
  sigprocmask(SIG_BLOCK, &full, NULL);


  childPid = fork();
  if (!childPid)
    sender(getppid(), argc, argv);
  else {
    father(childPid);
  }
}
