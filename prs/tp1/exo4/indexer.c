#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>

#define SUFFIXE ".idx"
#define BUF_SIZE 2048

void verifier(int cond, char *s) {
  if (!cond) {
    perror(s);
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char *argv[]) {
  verifier(argc == 2, "il faut un paramètre.");

  // construire le chemin au fichier index
  int l = strlen(argv[1]);
  char idx_filename[l + strlen(SUFFIXE) + 1];
  strncpy(idx_filename, argv[1], l);
  strcpy(idx_filename + l, SUFFIXE);

  // Open file for index
  int fdIn = open(argv[1], O_RDONLY);
  if (fdIn == -1) {
    fprintf(stdout, "Cannot open %s file\n", argv[1]);
    return EXIT_FAILURE;
  }

  // Create index file
  int fdOut = open(idx_filename, O_WRONLY | O_CREAT, 0666);
  if (fdOut == -1) {
    fprintf(stdout, "Cannot open %s file\n", idx_filename);
    close(fdIn);
    return EXIT_FAILURE;
  }

  while (1) {
    char value;
    // read one char and move cursor
    int n = read(fdIn, &value, sizeof(char));
    if (!n) break;

    if (value != '\n') continue;
    // if char is \n get current pos and write it in indexfile
    off_t pos = lseek(fdIn, 0, SEEK_CUR);
    write(fdOut, &pos, sizeof(off_t));
  }

  close(fdIn);
  close(fdOut);

  return EXIT_SUCCESS;
}
