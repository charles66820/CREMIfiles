#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char const *argv[]) {
  int child = fork();

  if (child) {
    printf("je m'appelle %d et je suis le père de %d\n", getpid(), child);
    wait(NULL); // for wait child is dead for not make zombie
    exit(0); // For make a zombie
  } else {
    //sleep(4); // Uncomment this line and comment wait(NULL) for see an orphan
    printf("je m'appelle %d et je suis le fils de %d\n", getpid(), getppid());
  }

  return EXIT_SUCCESS;
}
