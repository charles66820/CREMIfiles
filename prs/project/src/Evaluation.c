#include "Evaluation.h"

#include <fcntl.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "Shell.h"

int evaluer_expr(Expression *e) {
  if (e->type == VIDE) return 0;
  if (e->type == SIMPLE) {
    fork();
    
  }
  fprintf(stderr, "not yet implemented \n");
  return 1;
}
