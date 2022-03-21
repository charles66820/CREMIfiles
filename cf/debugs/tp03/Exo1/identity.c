#include "identity.h"

int identity_while(int n){
	int result = 0;
	while (result < n){
		result++;
	}
	return result;
}

int identity_for(int n){
	int result = 0;
	for (int i = 1; i <=n; i++){
		result++;
	}
	return result;
}

