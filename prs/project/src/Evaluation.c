#include "Evaluation.h"

#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

//#include <signal.h>

#include "Shell.h"

typedef unsigned int uint;
#define BUFSIZE 1000

static char** getArgs(int pid) {
  char cmdLineFile[24];
  sprintf(cmdLineFile, "/proc/%d/cmdline", pid);
  FILE* f = fopen(cmdLineFile, "r");
  if (f == NULL) {
    fprintf(stdout, "Cannot create %s file\n", cmdLineFile);
    exit(EXIT_FAILURE);
  }

  char** s = InitialiserListeArguments();
  while (!feof(f)) {
    char buff[BUFSIZE];
    if (fgets(buff, BUFSIZE - 1, f) != NULL) s = AjouterArg(s, buff);
  }
  fclose(f);
  return s;
}

int evaluer_expr(Expression* e) {
  int status;
  // Process exptression
  switch (e->type) {
    case VIDE:
      status = 0;
      break;
    case SIMPLE: {
      int argc = LongueurListe(e->arguments);
      if (!strcmp(e->arguments[0], "echo")) {
        for (uint i = 1; i < argc; i++) {
          printf("%s", e->arguments[i]);
          if (i == argc - 1)
            printf("\n");
          else
            printf(" ");
        }
        status = 0;
      } else {
        int pid;
        if (!(pid = fork())) {
          execvp(e->arguments[0], e->arguments);

          fprintf(stderr, "%s: command not found\n", e->arguments[0]);
          status = 127;
        }
        int exitStatus;
        waitpid(pid, &exitStatus, 0);
        status = WTERMSIG(exitStatus) ? WTERMSIG(exitStatus) + 128
                                      : WEXITSTATUS(exitStatus);
      }
      break;
    }
    case SEQUENCE: {
      evaluer_expr(e->gauche);
      status = evaluer_expr(e->droite);
      break;
    }
    case SEQUENCE_ET: {
      status = evaluer_expr(e->gauche);
      if (!status) status = evaluer_expr(e->droite);
      break;
    }
    case SEQUENCE_OU: {
      status = evaluer_expr(e->gauche);
      if (status) status = evaluer_expr(e->droite);
      break;
    }
    case BG: {
      if (!fork()) exit(evaluer_expr(e->gauche));
      status = 0;
      break;
    }

    default:
      fprintf(stderr, "not yet implemented\n");
      status = 1;
  }

  // Process backgrand process
  pid_t pid;
  int exitStatus;
  int count = 1;
  if (pid = waitpid(-1, &exitStatus, WNOHANG) > 0) {
    char* s = !WEXITSTATUS(exitStatus)
                  ? "Done"
                  : WEXITSTATUS(exitStatus) == 1
                        ? "Exit 1"
                        : WEXITSTATUS(exitStatus) > 128
                              ? strsignal(WEXITSTATUS(exitStatus) - 128)
                              : "Unkown exit code";
    printf("[%d]\t%s\n", count, s);
    count++;
  }
  return status;
}
