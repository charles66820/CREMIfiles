#ifdef CHANGED
#include "syscall.h"

const int N = 12;

int main() {
  PutString("Father proc!\n");

  ForkExec("test/userpages0");
  int i;
  for (i = 0; i < N; i++) ForkExec("test/userpages1");

  return 0;
}
#endif  // CHANGED