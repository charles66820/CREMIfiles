#include "writes.h"

int writes(int *tab, int size, int index, int value){
	if(index >= 0 && index < size){
		tab[index] = value;
		return 0;
	}
	return -1;
}