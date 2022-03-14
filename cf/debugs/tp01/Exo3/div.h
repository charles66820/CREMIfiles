#include "limits.h"

/*@
  requires b != 0;
  requires a / b <= INT_MAX;
  ensures \result == a/b;
*/
int div(int a, int b);