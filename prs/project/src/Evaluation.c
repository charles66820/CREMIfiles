#include "Evaluation.h"

#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "Shell.h"

typedef unsigned int uint;

int evaluer_expr(Expression *e) {
  switch (e->type) {
    case VIDE:
      return 0;
    case SIMPLE: {
      int argc = LongueurListe(e->arguments);
      if (!strcmp(e->arguments[0], "echo")) {
        for (uint i = 1; i < argc; i++) {
          printf("%s", e->arguments[i]);
          if (i == argc - 1) printf("\n");
          else printf(" ");
        }
      } else {
        if (!fork()) {
          execvp(e->arguments[0], e->arguments);

          fprintf(stderr, "%s: command not found\n", e->arguments[0]);
          return 127;
        }
        int exitCode;
        wait(&exitCode);
        return WTERMSIG(exitCode)? WTERMSIG(exitCode) + 128 : WEXITSTATUS(exitCode);
      }
      return 0;
    }
    case SEQUENCE: {
      evaluer_expr(e->gauche);
      return evaluer_expr(e->droite);
    }
    case SEQUENCE_ET: {
      int status = evaluer_expr(e->gauche);
      if (!status) return evaluer_expr(e->droite);
      return status;
    }
    case SEQUENCE_OU: {
      int status = evaluer_expr(e->gauche);
      if (!status) return status;
      return evaluer_expr(e->droite);
    }

    default:
      fprintf(stderr, "not yet implemented \n");
      return 1;
  }
}
