#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char* argv[]) {
  if (argc <= 1) {
    printf("Usage : %s <cmds> ... ", argv[0]);
    return EXIT_FAILURE;
  }

  printf("*** execution\n");


  int pid = fork();

  if (!pid) {
    execvp(argv[1], argv + 1);
    perror("error");
    return EXIT_FAILURE;
  }

  int exitCode;
  wait(&exitCode);

  printf("*** code de retour : %d\n", WEXITSTATUS(exitCode));

  return EXIT_SUCCESS;
}
