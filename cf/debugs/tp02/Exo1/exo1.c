#include "exo1.h"

int max_ptr(int *a, int *b){
	if (*a >= *b) return *a;
	return *b;
}

int sum_first_last(int *a, int n){
	return a[0] + a[n-1];
}

void swap_first_last(int *a, int n){
	int b = a[0];
	a[0] = a[n-1];
	a[n-1] = b;
	return;
}

void swap(int *a, int *b){
	int c = *a;
	*a = *b;
	*b = c;
	return;
}