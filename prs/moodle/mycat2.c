#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define BUFSIZE TMP_MAX

void redirFile2Stdout(int fd) {
  char buffer[BUFSIZE + 1];
  int n;
  while (1) {
    n = read(fd, buffer, BUFSIZE);
    if (!n) return;
    write(STDOUT_FILENO, buffer, n);
  };
}

int main(int argc, char* argv[]) {
  if (argc <= 1) redirFile2Stdout(STDIN_FILENO);
  for (unsigned int i = 1; i < argc; i++) {
    int fd = open(argv[i], O_RDONLY);
    if (fd == -1) {
      printf("Error on open : %s\n", argv[i]);
      continue;
    }
    redirFile2Stdout(fd);
    close(fd);
  }

  return EXIT_SUCCESS;
}
