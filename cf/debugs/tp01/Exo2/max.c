#include "max.h"

int max(int a, int b)
{
	int res;
	if (a >= b)
		res = a;
	else
		res = b;
	return res;
}