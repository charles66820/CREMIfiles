#include "limits.h"

/*@ requires a < INT_MAX;
    behavior div:
        assumes a <= 0;
        requires b != 0;
        ensures \result == a/b;

    behavior add:
        assumes a > 0 && b <= 0;
        ensures \result == a+b;

    behavior sub:
        assumes a > 0 && b >= 0;
        ensures \result == a-b;

    complete behaviors;
    disjoint behaviors div,add;
*/
int addOrDiv(int a, int b)
{
    if (a <= 0)
        return a / b;
    if (b <= 0)
        return a + b;
    return a - b;
}