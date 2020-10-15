#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char* argv[]) {
  if (argc <= 0) {
    printf("Usage : %s ... ", argv[0]);
    return EXIT_FAILURE;
  }

  //printf("*** execution\n");

  int p1[2];
  pipe(p1);

  write(p1[0], "c", sizeof(char));
  write(p1[0], "o", sizeof(char));
  write(p1[0], "u", sizeof(char));
  write(p1[0], "c", sizeof(char));
  write(p1[0], "o", sizeof(char));
  write(p1[0], "u", sizeof(char));

  int n;
  do {
    char c;
    n = read(p1[1], &c, sizeof(char));
    if (n) write(1, &c, sizeof(char));
  } while(!n);

  //int pid = fork();

  //if (!pid) {

    //perror("error");
    //return EXIT_FAILURE;
  //}

  //int exitCode;
  //wait(&exitCode);

  //printf("*** code de retour : %d\n", WEXITSTATUS(exitCode));

  return EXIT_SUCCESS;
}
