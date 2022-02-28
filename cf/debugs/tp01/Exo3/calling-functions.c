#include "plus_one.h"
#include "div.h"

int good_call_1(void)
{
    return plus_one(4);
}

int bad_call_1(void)
{
    return plus_one(-345);
}

int bad_call_2(void)
{
    return plus_one(INT_MAX);
}

int good_call_2(void)
{
    return div(45, 73);
}

int good_call_3(void)
{
    return div(0, 74);
}

int bad_call_3(void)
{
    return div(74, 0);
}

int good_call_4(void)
{
    return div(INT_MAX, INT_MIN);
}

int good_call_5(void)
{
    return div(INT_MAX, -1);
}

int bad_call_4(void)
{
    return div(INT_MIN, -1);
}