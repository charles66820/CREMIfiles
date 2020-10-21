#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char* argv[]) {
  if (argc < 3) {
    printf("Usage : %s <logfile> cmd [args...]", argv[0]);
    return EXIT_FAILURE;
  }

  // create pipe
  int pipe1[2];
  pipe(pipe1);

  int pidCmdRunner = fork();
  if (!pidCmdRunner) {
    // cmd runner
    // redirect cmd stdout to pipe
    dup2(pipe1[1], STDOUT_FILENO);
    execvp(argv[2], argv + 2);

    fprintf(stderr, "Error on exec cmd\n");
    return EXIT_FAILURE;
  } else {
    int pidLogWriter = fork();
    if (!pidLogWriter) {
      // log writer
      // close pipe write for this child
      close(pipe1[1]);
      // write cmd stdout form pipe to logfile and main process stdout with tee
      dup2(pipe1[0], STDIN_FILENO);
      execlp("tee", "tee", argv[1], (char*)NULL);

      fprintf(stderr, "Error on exec tee\n");
      return EXIT_FAILURE;
    } else {
      // on cmd finish close pipe stdin
      int status;
      waitpid(pidCmdRunner, &status, 0);
      close(pipe1[1]);

      // wait for log writer
      wait(NULL);
      return WEXITSTATUS(status);
    }
  }
}
