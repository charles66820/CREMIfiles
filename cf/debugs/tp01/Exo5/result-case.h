/*@
  ensures \result >= 0 || \result <= 3;
  ensures a==b || b==c || a==c ==> \result == 0; // \result == 0 when a==b || b==c || a==c
  ensures a < b < c || a < c < b ==> \result == 1; // \result == 1 when a < b&&c
  ensures b < a < c || b < c < a ==> \result == 2; // \result == 2 when b < a&&c
  ensures c < a < b || c < b < a ==> \result == 3; // \result == 3 when c < a&&b
*/
int caseResult(int a, int b, int c);