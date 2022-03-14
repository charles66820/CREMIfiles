/*@
  predicate smallerThen(integer a, integer b, integer c) = a < b < c || a < c < b; // a < b&&c
*/

/*@
  ensures \result >= 0 || \result <= 3;
  ensures a==b || b==c || a==c ==> \result == 0; // \result == 0 when a==b || b==c || a==c
  ensures smallerThen(a, b, c) ==> \result == 1; // \result == 1 when a < b&&c
  ensures smallerThen(b, a, c) ==> \result == 2; // \result == 2 when b < a&&c
  ensures smallerThen(c, a, b) ==> \result == 3; // \result == 3 when c < a&&b
*/
int caseResult(int a, int b, int c);