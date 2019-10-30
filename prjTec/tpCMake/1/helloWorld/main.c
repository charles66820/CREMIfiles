#include <stdio.h>
#include "foo.h"
#include "bar.h"

int main(int argc, char const *argv[])
{
	int i1 = foo_1();
	int i2 = foo_2();
	double x = bar(i1);
	printf(" foo 1= %d,  foo 2 = %d, bar = %g\n", i1, i2, x);
	return 0;
}