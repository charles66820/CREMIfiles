#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <fcntl.h>
#include <stdio.h>

int main(int argc, char* argv[]) {

  if (argc != 4) return EXIT_FAILURE;

  int fdOut = open (argv[1], O_WRONLY | O_CREAT, 0666);
  if (fdOut == -1) {
    fprintf(stdout, "Cannot open %s file\n", argv[1]);
    return EXIT_FAILURE;
  }

  u_int32_t value = atoi(argv[3]);

  // position
  lseek(fdOut, atoi(argv[2]) * sizeof(u_int32_t), SEEK_SET);
  // si on cherche à écrire au-delà de la fin du fichier le fichier s'agrandi mais ne complete pas l'intervale avec des 0 il compte juste l'espace comme utiliser

  write(fdOut, &value, sizeof(u_int32_t));

  close(fdOut);

  return EXIT_SUCCESS;
}
