#include "limits.h"

/*@
  requires a * x >= INT_MIN;
  requires a * x <= INT_MAX;
  requires (a * x) + b >= INT_MIN;
  requires (a * x) + b <= INT_MAX;
  ensures \result == a * x + b;
*/
int f(int a,int b,int x);