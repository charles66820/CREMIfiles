#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char* argv[]) {
  // pipe
  int pipe1[2];
  pipe(pipe1);

  int n;
  double t = 0;

  do {
    n = write(pipe1[1], "k", sizeof(char)); // block here after 65536 char = 65.536 ko
    t++;
    printf("%f time\n", t);
  } while(n);

  close(pipe1[1]);
  return EXIT_SUCCESS;
}
