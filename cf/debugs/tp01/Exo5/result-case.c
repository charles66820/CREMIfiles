#include "result-case.h"

int caseResult(int a, int b, int c)
{
    if (a == b || b == c || a == c)
        return 0;
    if (a <= b && a <= c)
        return 1;
    if (b <= a && b <= c)
        return 2;
    if (c <= a && c <= b)
        return 3;
    return 2;
}