#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char* argv[]) {
  if (argc <= 0) {
    printf("Usage : %s ... ", argv[0]);
    return EXIT_FAILURE;
  }

  int p1[2];
  pipe(p1);

  write(p1[1], "coucou", sizeof(char) * 6);

  int n;
  do {
    char c;
    n = read(p1[0], &c, sizeof(char));
    if (n) write(STDOUT_FILENO, &c, sizeof(char));
  } while(n);

  return EXIT_SUCCESS;
}
