#ifdef CHANGED
#include "syscall.h"

int main() {
  PutString("Father proc!\n");

  ForkExec("test/putchar");
  ForkExec("test/putchar");

  // while (1);
  return 0;
}
#endif  // CHANGED