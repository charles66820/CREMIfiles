#include "limits.h"

/*@
  requires \valid(a);
  requires \valid(b);
  ensures *a <= *b;
  ensures (\old(*a) == *a && \old(*b) == *b) || (\old(*a) == *b && \old(*b) == *a);
*/
void sort_ptr(int* a, int* b);

/*@
  requires \valid(a);
  requires \valid_read(b);
  requires \separated(a, b);
  requires *a + *b >= INT_MIN;
  requires *a + *b <= INT_MAX;
  ensures *a == \old(*a) + \old(*b);
  ensures \old(*b) == *b;
*/
void sum_in_pointer(int* a, int* b);