#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

const char* TMP_FILE_NAME = "tmp.pipe";

int main(int argc, char* argv[]) {
  // args cmd1 cmd2 args...
  if (argc < 3) {
    printf("Usage : %s cmd1 cmd2 [args...]", argv[0]);
    return EXIT_FAILURE;
  }

  // tmp file rw
  int fd = open(TMP_FILE_NAME, O_RDWR | O_CREAT, 0666);
  if (fd == -1) {
    fprintf(stdout, "Cannot create %s file\n", TMP_FILE_NAME);
    return EXIT_FAILURE;
  }

  // cmd1 1 > 0 cmd2 args...
  if (!fork()) {
    // Change cmd1 stdout
    dup2(fd, STDOUT_FILENO);
    execlp(argv[1], argv[1], (char*)NULL);

    fprintf(stderr, "Error on exec cmd1\n");
    return EXIT_FAILURE;
  } else {
    // on cmd1 finish
    wait(NULL);

    // go to file begin
    lseek(fd, 0, SEEK_SET);

    // execute cmd2
    if (!fork()) {
      // Change cmd2 stdout
      dup2(fd, STDIN_FILENO);
      execvp(argv[2], argv + 2);

      fprintf(stderr, "Error on exec cmd2\n");
      return EXIT_FAILURE;
    } else {
      // on cmd2 finish
      wait(NULL);

      if (remove(TMP_FILE_NAME)) fprintf(stderr, "Error on delete temp file\n");

      close(fd);
      return EXIT_SUCCESS;
    }
  }
}
