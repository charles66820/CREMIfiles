#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <fcntl.h>
#include <stdio.h>

int main(int argc, char* argv[]) {

  int fdErr = open("ERREURS-LIRE.log", O_WRONLY | O_CREAT, 0666);
  if (fdErr == -1) {
    fprintf(stdout, "Cannot create ERREURS-LIRE.log file\n");
    return EXIT_FAILURE;
  }

  // redirect stderr to fdErr file
  dup2(fdErr, STDERR_FILENO);

  if (argc != 3) {
    fprintf(stdout, "Usage : %s <filename> <position>\n", argv[0]);
    return EXIT_FAILURE;
  }

  int fdIn = open(argv[1], O_RDONLY);
  if (fdIn == -1) {
    fprintf(stdout, "Cannot open %s file\n", argv[1]);
    return EXIT_FAILURE;
  }

  // position
  int position;
  if (argv[2][0] == '0') position = 0;
  else {
    position = atoi(argv[2]);
    if (position == 0) {
      fprintf(stdout, "position need to be a number\n");
      return EXIT_FAILURE;
    }
  }

  lseek(fdIn, position * sizeof(u_int32_t), SEEK_SET);

  u_int32_t value;
  int n = read(fdIn, &value, sizeof(u_int32_t));
  close(fdIn);
  if (!n) {
   fprintf(stderr, "error on red value");
   return EXIT_FAILURE;
  }
  // Si pm cherche à lire au-delà de la fin du fichier on ne peut pas lire

  char buffer[n];
  sprintf(buffer, "%u\n", value);

  write(STDOUT_FILENO, buffer, n);

  return EXIT_SUCCESS;
}
