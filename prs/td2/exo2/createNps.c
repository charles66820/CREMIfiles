#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char const *argv[]) {
  if (argc != 2) {
    printf("Usage : %s <nbChild>", argv[0]);
    return EXIT_FAILURE;
  }

  for (int i = 0; i < atoi(argv[1]); i++) {
    int childPid = fork();
    if (!childPid) {
      // If child exit
      printf("je m'appelle %d et je suis le fils de %d le %d eme\n", getpid(),
             getppid(), i);
      return EXIT_SUCCESS;
    } else {
      printf("je m'appelle %d et je suis le père de %d mon %d eme\n", getpid(),
             childPid, i);
    }
  }

  for (int i = 0; i < atoi(argv[1]); i++) {
    wait(NULL);
  }
  return EXIT_SUCCESS;
}
