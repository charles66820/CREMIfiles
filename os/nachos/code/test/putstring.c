#ifdef CHANGED
#include "syscall.h"

int main() {
  char s[] = "A small test string!\n";
  PutString(s);
  char sl[] = // 21 char + 128 '1' char
      "A long test string!"
      "111111111111111111111111111111111111111111111111111111111111111111111111"
      "11111111111111111111111111111111111111111111111111111111\n";
  PutString(sl);
  Halt();
}
#endif  // CHANGED