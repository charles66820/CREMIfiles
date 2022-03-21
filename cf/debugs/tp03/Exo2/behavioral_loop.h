#include "limits.h"

/*@ requires 0 < n;
  @ requires 2*n <= INT_MAX;
  @ 
  @ behavior cPos:
  @ 	assumes c >= 0;
  @ 	ensures \result == 2*n;
  @ 
  @ behavior cNeg:
  @ 	assumes c < 0;
  @ 	ensures \result == n;
  @
  @ complete behaviors;
  @ disjoint behaviors;
*/
int behavioral_loop(int n, int c);