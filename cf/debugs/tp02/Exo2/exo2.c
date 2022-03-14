#include "exo2.h"

void sort_ptr(int* a, int* b){
	if(*a > *b){
	int c = *a;
	*a = *b;
	*b = c;
	}
}

void sum_in_pointer(int *a, int *b){
	*a += *b;
}
