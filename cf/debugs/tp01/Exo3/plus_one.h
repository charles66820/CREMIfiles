#include "limits.h"

/*@
  requires a >= -1;
  requires a < INT_MAX;
  ensures \result >= 0;
*/
int plus_one(int a);