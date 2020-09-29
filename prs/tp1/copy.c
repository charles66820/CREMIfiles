#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <fcntl.h>
#include <stdio.h>

int main(int argc, char* argv[]) {

  if (argc != 3 && argc != 4) return EXIT_FAILURE;

  int fdIn = open(argv[1], O_RDONLY);
  if (fdIn == -1) {
    printf("Cannot open %s file\n", argv[1]);
    return EXIT_FAILURE;
  }

  int fdOut = open(argv[2], O_WRONLY + O_CREAT, 0664); //O_SYNC
  if (fdOut == -1) {
    printf("Cannot open %s file\n", argv[2]);
    close(fdIn);
    return EXIT_FAILURE;
  }

  int bufferSize = 1;

  if (argc == 4) bufferSize += atoi(argv[3]);
  char buffer[bufferSize];
  int n;

loop:
  n = read(fdIn, buffer, 1);
  if (!n) {
    close(fdIn);
    close(fdOut);
    return EXIT_SUCCESS;
  }
  write(fdOut, buffer, n);
  goto loop;
}
