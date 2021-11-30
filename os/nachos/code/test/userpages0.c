#ifdef CHANGED
#include "syscall.h"

int main() {
  char s[] = "Hello world !\n";
  PutString(s);

  char test[] = "Test avec plusieurs petits char !\n";
  PutString(test);
  char c = 'a';
  int i;
  for (i = 0; i < 5; i++) {
      PutChar(c + i);
  }
  PutChar('\n');

  return 0;
}
#endif  // CHANGED