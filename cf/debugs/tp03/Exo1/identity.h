#include "limits.h"

/*@ requires n > 0;
  @ requires n < INT_MAX;
  @ ensures \result == n;
*/
int identity_while(int n);

/*@ requires n > 0;
  @ requires n < INT_MAX;
  @ ensures \result == n;
*/
int identity_for(int n);
