#include "exo4.h"

int max_element(const int* t, int n){
	int max = 0;
	for (int i = 0; i < n; i++){
		if (t[max] < t[i]) max = i;
	}
	return max;
}

void sum_of_tab(int *a, int *b, int n){
	for (int i = 0; i<n; i++){
		a[i] += b[i];
	}
}