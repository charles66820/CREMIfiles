#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// Without optimization `int i` work but with optimization `int i` generate a loop becaue i is not in RAM.
// With `volatile int i` the program work

jmp_buf buf;

void f(int i) {
  longjmp(buf, i); // Warning i is copy
}

int g() {
  return setjmp(buf); // this break all
}

int main(int argc, char *argv[]) {
  volatile int i = 0;
  if(g() < 10) {
    printf("%d\n", i);
    f(++i);
  }
  return EXIT_SUCCESS;
}
