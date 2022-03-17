#include "limits.h"

/*@
  requires \valid(a);
  requires *a + 1 <= INT_MAX;
  assigns *a;
  ensures *a == \old(*a + 1);
*/
void add_one(int *a);

/*@
  assigns \nothing;
*/
void dummy();

/*@
  requires \valid(a) && \valid(b);
  requires *a+1 <= INT_MAX;
  requires \separated(a, b);
  ensures \result == 1;
  ensures *a == \old(*a+1);
  ensures *b == \old(*b);
*/
int f(int *a, int *b);