#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define SIZE 10000000
extern void dep(float *x,float *A);

// gcc -o dep dep.c depmain.c -lm

int main() {
  int i,j;
  float *A;
  float x=0;

  A= (float*)valloc(sizeof(float)*SIZE);
  for(i=0;i<SIZE;i++) A[i]=i+1;

  dep(&x,A);

  free(A);

  printf("%f\n",x);
  return 0;
}