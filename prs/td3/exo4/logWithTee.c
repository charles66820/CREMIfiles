#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char* argv[]) {
  // args cmd1 cmd2 args...
  if (argc < 3) {
    printf("Usage : %s logfile cmd [args...]", argv[0]);
    return EXIT_FAILURE;
  }

  int pid = fork();
  if (!pid) {
    char s[TMP_MAX];
    sprintf(s, "%s | tee %s", argv[argc-1], argv[1]);
    argv[argc-1] = s;
    execvp(argv[2], argv + 2);

    fprintf(stderr, "Error on exec cmd\n");
    return EXIT_FAILURE;
  } else {
    int status;
    waitpid(pid, &status, 0);
    return WEXITSTATUS(status);
  }
}
