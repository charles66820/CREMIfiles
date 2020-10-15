#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char* argv[]) {
  // args cmd1 cmd2 args...
  if (argc < 3) {
    printf("Usage : %s cmd1 cmd2 [args...]", argv[0]);
    return EXIT_FAILURE;
  }

  // pipe
  int pipe1[2];
  pipe(pipe1);

  // cmd1 1 > 0 cmd2 args...
  if (!fork()) {
    // Change cmd1 stdout
    dup2(pipe1[1], STDOUT_FILENO);
    execlp(argv[1], argv[1], (char*)NULL);

    fprintf(stderr, "Error on exec cmd1\n");
    return EXIT_FAILURE;
  } else {
    // on cmd1 finish
    int exitCode;
    wait(&exitCode);
    close(pipe1[1]); // Imperative call because pipe need end for cmd2 can be close

    // execute cmd2
    if (!fork()) {// this child not close if pipe stdin are not close
      // Change cmd2 stdout
      dup2(pipe1[0], STDIN_FILENO);
      execvp(argv[2], argv + 2);

      fprintf(stderr, "Error on exec cmd2\n");
      return EXIT_FAILURE;
    } else {
      // on cmd2 finish
      wait(exitCode? NULL : &exitCode);
      return WEXITSTATUS(exitCode);
    }
  }

  return 0;
}
