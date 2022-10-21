#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define SIZE 10000000

void dep(float *x,float *A) {
  int i,j;
  for (j=0; j<100; j++)
    for (i=0; i<SIZE; i++) {
      *x=*x+1./sqrt(A[i]);
    }
}
