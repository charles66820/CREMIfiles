#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char* argv[]) {
  // args cmd1 cmd2 args...
  if (argc < 3) {
    printf("Usage : %s logfile cmd [args...]", argv[0]);
    return EXIT_FAILURE;
  }

  // open log file
  int fd = open(argv[1], O_WRONLY | O_CREAT, 0666);
  if (fd == -1) {
    fprintf(stdout, "Cannot create %s file\n", argv[1]);
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
      // write cmd stdout form pipe to logfile and main process stdout
      int n;
      char c;
      do {
        n = read(pipe1[0], &c, sizeof(char));
        if (n) {
          write(STDOUT_FILENO, &c, sizeof(char));
          write(fd, &c, sizeof(char));
        }
      } while (n);
      return EXIT_SUCCESS;
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
