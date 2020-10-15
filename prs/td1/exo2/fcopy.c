#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char* argv[]) {
  if (argc != 3) return EXIT_FAILURE;

  FILE* fIn = fopen(argv[1], "r");
  if (!fIn) {
    fprintf(stdout, "Cannot open %s file\n", argv[1]);
    return EXIT_FAILURE;
  }

  FILE* fOut = fopen(argv[2], "w");
  if (!fOut) {
    fprintf(stdout, "Cannot open %s file\n", argv[2]);
    fclose(fIn);
    return EXIT_FAILURE;
  }

  char buffer[1];
  int n;

loop:
  n = fread(buffer, sizeof(char), 1, fIn);
  if (!n) {
    fclose(fIn);
    fclose(fOut);
    return EXIT_SUCCESS;
  }
  fwrite(buffer, sizeof(char), n, fOut);
  goto loop;
}
