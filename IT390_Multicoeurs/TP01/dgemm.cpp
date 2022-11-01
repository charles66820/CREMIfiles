#include <stdio.h>
#include <stdlib.h>
#include "mipp.h"

#define N 206 // M width
#define SIZE N*N // M total mem SIZE : 42436

// 3 Matrix mem SIZE : 127308

typedef float type;

// g++ -I/home/charles/github/MIPP/src -o dgemm dgemm.cpp

void dgemm(type* A, type* B, type* C) {
  size_t i, j, k;

  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      for (k = 0; k < N; k++) C[i * N + j] += A[i * N + k] * B[k * N + j];
}

static mipp::Reg<type> v1;
static mipp::Reg<type> v2;

/**
 * @brief dgemm
 *
 * @param A fist matrix
 * @param B second matrix /!\ need to be transposed
 * @param C Result matrix
 */
void dgemm_MIPP(type* A, type* B, type* C) {
  size_t i, j, k;

  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++) {
        type tmp = 0;

        for(int n = 0; n < N; n += mipp::nElReg<type>()) {
          v1.load(&A[i * N + n]);
          v2.load(&B[i * N + n]);

          auto mul_res = v1 * v2;
          tmp += mipp::Reduction<type, mipp::add>::sapply(mul_res);
        }
        C[i * N + j] = tmp;
      }
}

void printM(type* M) {

  printf("[");
  for (size_t i = 0; i < SIZE; i++) {
    if (i != 0) printf(", ");
    printf("%f", M[i]);
  }
  printf("]\n");
}

int main(int argc, char const* argv[]) {
  type* A = (type*)valloc(SIZE * sizeof(double));
  type* B = (type*)valloc(SIZE * sizeof(double));
  type* Bt = (type*)valloc(SIZE * sizeof(double));
  type* C = (type*)valloc(SIZE * sizeof(double));
  if (A == NULL || B == NULL || C == NULL) exit(EXIT_FAILURE);

  for (size_t i = 0; i < SIZE; i++) {
    A[i] = i;
    B[i] = i + 1;
    C[i] = 0;
  }

  dgemm(A, B, C);
  printM(C);

  for (size_t i = 0; i < N; ++i)
    for (size_t j = 0; j < N; ++j)
        Bt[j * N + i] = B[i * N + j];

  dgemm_MIPP(A, Bt, C);

  // printM(C);

  free(A);
  free(B);
  free(C);

  return EXIT_SUCCESS;
}
