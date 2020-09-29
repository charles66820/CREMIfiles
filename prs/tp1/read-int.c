#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <fcntl.h>
#include <stdio.h>

int main(int argc, char* argv[]) {

  if (argc != 3) return EXIT_FAILURE;

  int fdIn = open(argv[1], O_RDONLY);
  if (fdIn == -1) {
    printf("Cannot open %s file\n", argv[1]);
    return EXIT_FAILURE;
  }

  // position
  lseek(fdIn, atoi(argv[2]) * 2, SEEK_SET);

  u_int32_t value;
  int n = read(fdIn, &value, 2);
  close(fdIn);
  if (!n) return EXIT_FAILURE;
  // Si pm cherche à lire au-delà de la fin du fichier on ne peut pas lire

  char buffer[n];
  sprintf(buffer, "%d\n", value);

  write(STDOUT_FILENO, buffer, n);

  return EXIT_SUCCESS;
}
