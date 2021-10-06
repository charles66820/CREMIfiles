#ifdef CHANGED
#include "syscall.h"

int main() {
  char s[] = "A small test string!\n";
  PutString(s);
  Halt();
}
#endif  // CHANGED