#include <stdio.h>
#include <stdlib.h>

int main() {
	int tab[10];
	unsigned int i; // CORRECTION : il manque la déclaration de la variable i

	tab[0] = 1; // inisialisation de la premier case du tableaux à 1

	// on navique dans le tableaux de 1 a n-1 (de 1 a 10)
	// on multiplie le nombre de la case présédante par 2 pour le mettre le résultat dans la case cibler par i
	for(i = 1; i <= 10; ++i) { // CORRECTION : on remplace le "i < 10" par "i <= 10"
		tab[i] = tab[i-1] * 2;
	}

	// affiche le contenu du tableaux
	for(i = 0; i < 11; ++i) {
		printf("%d ", tab[i]);
	}
	printf("\n");

	return EXIT_SUCCESS;
}
