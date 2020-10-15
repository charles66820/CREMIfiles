#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char* argv[]) {
  // pipe
  int pipe1[2];
  pipe(pipe1);
  close(pipe1[1]);
  char c;
  int n = read(pipe1[0], &c, sizeof(char));
  printf("read (%d) %c in pipe1\n", n, c); // 1. the pipe is not blocking

  int pipe2[2];
  pipe(pipe2);
  close(pipe2[0]);
  char c2 = 'k';
  n = write(pipe2[1], &c2, sizeof(char)); // 2. exit with code 141 (128 + 13) SIGPIPE Broken pipe (POSIX).
  printf("write (%d) %c in pipe2\n", n, c2);
  return EXIT_SUCCESS;
}
