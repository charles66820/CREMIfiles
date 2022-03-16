/*@
  predicate smallerThen(integer a, integer b, integer c) = a < b < c || a < c < b; // a < b&&c
*/

/*@
  ensures \result >= 0 || \result <= 3;

  behavior twoEq:
    assumes a==b || b==c || a==c; // when a==b || b==c || a==c
    ensures \result == 0;
  behavior smallA:
    assumes smallerThen(a, b, c); // when a < b&&c
    ensures \result == 1;
  behavior smallB:
    assumes smallerThen(b, a, c); // when b < a&&c
    ensures \result == 2;
  behavior smallC:
    assumes smallerThen(c, a, b); // when c < a&&b
    ensures \result == 3;

  complete behaviors;
  disjoint behaviors twoEq,smallA,smallB,smallC;
*/
int caseResult(int a, int b, int c);