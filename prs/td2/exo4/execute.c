#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

int pid;

int execute(char** argv) {
  pid = fork();

  if (!pid) {

    execvp(argv[0], argv);

    int exitCode;
    int thisPid = wait(&exitCode);
    if (thisPid == -1) return EXIT_FAILURE;

    return exitCode;
  }

  int exitCode;
  waitpid(pid, &exitCode, 0);

  return exitCode;
}

int main(int argc, char* argv[]) {
  if (argc <= 1) {
    printf("Usage : %s <cmds> ... ", argv[0]);
    return EXIT_FAILURE;
  }

  printf("*** execution\n");

  int code = execute(argv + 1);

  if (pid) printf("*** code de retour : %d\n", code);

  return EXIT_SUCCESS;
}
