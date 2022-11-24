#include <stdio.h>
#include <stdlib.h>

#include <chrono>

#include "mipp.h"

#define N 208      // M width 206
#define SIZE N* N  // M total mem SIZE : 42436

// 3 Matrix mem SIZE : 127308

typedef float type;

static std::chrono::time_point<std::chrono::system_clock> start, end;
#define TIMER_START() start = std::chrono::system_clock::now();

#define TIMER_END(msg)                                                        \
  end = std::chrono::system_clock::now();                                     \
  {                                                                           \
    std::chrono::duration<double, std::milli> elapsed_seconds = end - start;  \
    std::cout << msg " in " << elapsed_seconds.count() << " ms" << std::endl; \
  }

// c++ -I/home/charles/github/MIPP/src -std=c++11 -finline -march=native -o
// dgemm dgemm.cpp

// c++ -I/home/cisd-goedefr/projects/MIPP/src -std=c++11 -finline -march=native
// -o dgemm dgemm.cpp

void dgemm(type* A, type* B, type* C) {
  size_t i, j, k;

  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      for (k = 0; k < N; k++) {
        C[i * N + j] += A[i * N + k] * B[k * N + j];
      }
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

      for (int n = 0; n < N; n += mipp::nElReg<type>()) {
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

void printXM(type* M, int X) {
  for (size_t i = 0; i < X; i++)
    for (size_t j = 0; j < X; j++) {
      if (j != 0)
        std::cout << ", ";
      else
        std::cout << "[ ";
      std::cout << M[i * N + j];
      // printf("%f", M[i * N + j]);
      if (j == X-1) std::cout << ", ... ]" << std::endl;
    }
}

int main(int argc, char const* argv[]) {
  type* A = (type*)valloc(SIZE * sizeof(type));
  type* B = (type*)valloc(SIZE * sizeof(type));
  type* Bt = (type*)valloc(SIZE * sizeof(type));
  type* C = (type*)valloc(SIZE * sizeof(type));
  type* C2 = (type*)valloc(SIZE * sizeof(type));
  if (A == NULL || B == NULL || C == NULL) exit(EXIT_FAILURE);

  for (size_t i = 0; i < SIZE; i++) {
    A[i] = i;
    B[i] = i + 1;
    C[i] = 0;
    C2[i] = 0;
  }

  TIMER_START()
  dgemm(A, B, C);
  TIMER_END("dgemm seq")

  // transpose
  for (size_t i = 0; i < N; ++i)
    for (size_t j = 0; j < N; ++j) Bt[j * N + i] = B[i * N + j];

  TIMER_START()
  dgemm_MIPP(A, Bt, C2);
  TIMER_END("dgemm seq MIPP")

  for (size_t i = 0; i < SIZE; i++)
    if (C[i] != C2[i]) {
      std::cout << "C and C2 differ" << std::endl;
      printXM(C, 4);
      std::cout << std::endl;
      printXM(C2, 4);
      break;
    }

  free(A);
  free(B);
  free(C);

  return EXIT_SUCCESS;
}
