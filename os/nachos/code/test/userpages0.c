#ifdef CHANGED
#include "syscall.h"

int main() {
  char s[] = "Hello world !\n";
  char test[] = "Test avec plusieurs petits char !\n";
  char c = 'a';
  int i;

  PutString(s);
  PutString(test);
  for (i = 0; i < 5; i++) {
      PutChar(c + i);
  }
  PutChar('\n');

  return 0;
}
#endif  // CHANGED