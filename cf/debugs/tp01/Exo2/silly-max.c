#include "max.h"

int max(int a, int b)
{
    if (a == 0 && b == 0)
        return 0;
    if (a <= 0 && b >= 0)
        return b;
    if (a > 42 && b <= 25)
        return a;
    if (a >= 0 && b <= 0)
        return a;
    if (a - b > 4)
        return a;
    if (a - b < 1)
        return b;
    if (a == b + 1)
        return a;
    if (a <= b + 4)
        if (a + 1 < b)
            return b;
        else
            return a;
    return 42;
}