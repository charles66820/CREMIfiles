#include <stdio.h>
#include <stdlib.h>
 
int main() {
	int tab[10];
	
	tab[0] = 1;
	for(i = 1; i < 10; ++i) {
		tab[i] = tab[i-1] * 2;
	}
 
	for(i = 0; i < 11; ++i) {
		printf("%d ", tab[i]);
	}
	printf("\n");
 
	return EXIT_SUCCESS;
}