#include "limits.h"

/*@
  requires a * x >= -2147483648;
  requires a * x <= 2147483647;
  requires (a * x) + b >= -2147483648;
  requires (a * x) + b <= 2147483647;
  ensures \result == a * x + b;
*/
int f(int a,int b,int x);