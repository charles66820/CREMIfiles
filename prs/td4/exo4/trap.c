#include <signal.h>
#include <stdio.h>

volatile char *a = NULL;
volatile char b = 'x';

void traitant(int s) {
  printf("signal %d\n", s);
  a = &b;
}

int main() {
  struct sigaction s;
  char x;

  s.sa_handler = traitant;
  sigemptyset(&s.sa_mask);
  s.sa_flags = 0;
  sigaction(SIGSEGV, &s, NULL);

  x = *a;

  printf("fin %c\n", x);
  return 0;
}

// Le programme à faire une boucle infinie car en assembleur l'instruction x = *a se fait en plusieurs instructions donc si on change en mémoire la valeur du pointeur cette valeur ne sera pas recharger dans un registre et l'instruction qui lit grâce au registre dans la mémoire au mauvaise endroit continuera à planter elle ne sera pas rectifié.
