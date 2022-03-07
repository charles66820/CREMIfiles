#include "limits.h"

/*@
  requires b != 0;
  requires a / b <= 2147483647;
  ensures \result == a/b;
*/
int div(int a, int b);