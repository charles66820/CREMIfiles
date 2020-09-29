#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define BUFSIZE TMP_MAX

int main(int argc, char* argv[]) {
  int fdOut = 0;

  // If file
  if (argc > 1) {
    fdOut = open(argv[1], O_WRONLY | O_CREAT, 0664);
    if (fdOut == -1) {
      printf("Cannot open %s file\n", argv[1]);
      return EXIT_FAILURE;
    }
  }

  char buffer[BUFSIZE + 1];
  int n;

loop:
  n = read(STDIN_FILENO, buffer, BUFSIZE);
  if (!n) {
    if (fdOut) close(fdOut);
    return EXIT_SUCCESS;
  }
  write(STDOUT_FILENO, buffer, n);
  if (fdOut) write(fdOut, buffer, n);
  goto loop;
}
