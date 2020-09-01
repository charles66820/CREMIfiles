#include <stdio.h>
#include <stdlib.h>

#define BUFFER_IN TMP_MAX

void redirFile2Stdout(FILE* file) {
  char* p = calloc(1, BUFFER_IN);
  if (!p) printf("Not enough memory");
  while (!feof(file)) {
    fgets(p, BUFFER_IN, file);
    fputs(p, stdout);
    free(p);
    p = calloc(1, BUFFER_IN);
    if (!p) printf("Not enough memory");
  };
  free(p);
}

int main(int argc, char* argv[]) {
  if (argc <= 1) redirFile2Stdout(stdin);
  for (unsigned int i = 1; i < argc; i++) {
    FILE* file = fopen(argv[i], "r");
    if (file == NULL) {
      printf("Error on open : %s\n", argv[i]);
      continue;
    }
    redirFile2Stdout(file);
    fclose(file);
  }

  return EXIT_SUCCESS;
}
