#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

int System(char *commande) {
  int pid = fork();

  if (!pid) {
    execl("/bin/bash", "bash", "-c", commande, (char *)NULL);
    return EXIT_FAILURE;
  }
  wait(&pid);

  return 0;
}

int main(int argc, char const *argv[]) {
  int code = System("echo oui");

  return EXIT_SUCCESS;
}
