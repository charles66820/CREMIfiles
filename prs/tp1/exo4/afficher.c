#include <errno.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SUFFIXE ".idx"

void verifier(int cond, char *s){
  if (!cond){
    perror(s);
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char *argv[]){
  verifier(argc == 3, "il faut deux paramètres.");

  // construire le chemin au fichier index
  int l = strlen(argv[1]) ;
  char idx_filename[l + strlen(SUFFIXE) + 1];

  strncpy(idx_filename, argv[1], l);
  strcpy(idx_filename + l, SUFFIXE);

  // Open file
  int fdIn = open(argv[1], O_RDONLY);
  if (fdIn == -1) {
    fprintf(stdout, "Cannot open %s file\n", argv[1]);
    return EXIT_FAILURE;
  }

  // Open index file
  int fdInd = open(idx_filename, O_RDONLY);
  if (fdInd == -1) {
    fprintf(stdout, "Cannot open %s file\n", idx_filename);
    close(fdIn);
    return EXIT_FAILURE;
  }

  int line = atoi(argv[2]);
  if (line > 1) { // >= 1 if start at 0
    line -= 2; // -= 1 if start at 0

    // go to ligne position in index file
    lseek(fdInd, line * sizeof(off_t), SEEK_SET);

    // get file position in index file
    off_t value;
    int n = read(fdInd, &value, sizeof(off_t));
    if (!n) return EXIT_FAILURE;

    // go to position in file
    lseek(fdIn, value, SEEK_SET);
  }

  while (1) {
    char value;
    // read one char and move cursor
    int n = read(fdIn, &value, sizeof(char));
    if (!n) break;

    write(STDOUT_FILENO, &value, sizeof(char));

    if (value == '\n') break;
  }

  close(fdIn);
  close(fdInd);

  return EXIT_SUCCESS;
}
