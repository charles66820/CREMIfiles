#include "limits.h"

/*@ requires \valid(a);
  @ requires *a+1 <= INT_MAX;
  @ ensures *a == \old(*a+1);
*/
void add_one(int *a);

void dummy();

/*@ requires \valid(a) && \valid(b);
  @ requires *a+1 <= INT_MAX;
  @ ensures \result == 1;
  @ ensures *a == \old(*a+1);
  @ ensures *b == \old(*b);
*/
int f(int *a, int *b);