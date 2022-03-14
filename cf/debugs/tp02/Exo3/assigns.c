#include "assigns.h"

void add_one(int *a)
{
	*a += 1;
}

void dummy()
{
}

int f(int *a, int *b)
{
	int c = 1;
	add_one(a);
	//@ assert *a == \at(*a,Pre)+1;
	//@ assert *b == \at(*b,Pre);
	dummy();
	return c;
}
