#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>

#include "try.h"

void yes_or_no(void *reponse) {
  char c[100];
  read(0, c, 100);
  *(char *)reponse = (*c == 'o' || *c == 'O');
}

int main() {
  char rep;

  printf(" oui ou non ? \n");

  if (tryBefore(yes_or_no, &rep, 3) == 0)
    printf("reponse : %c\n", rep == 1 ? 'O' : 'N');
  else
    printf("pas de reponse\n");
}
