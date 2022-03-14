#include "limits.h"

/*@ requires 0 < size;
  @ requires \valid(tab + (0 .. size-1));
  @ 
  @ behavior sucess:
  @ 	assumes 0 <= index < size;
  @ 	assigns tab[index];
  @ 	ensures tab[index] == value;
  @ 	ensures \result == 0;
  @
  @ behavior outOfBounds:
  @ 	assumes index < 0 || size <= index;
  @ 	assigns \nothing;
  @ 	ensures \result == -1;
  @
  @ disjoint behaviors;
  @ complete behaviors;
*/
int writes(int *tab, int size, int index, int value);