#include <stdio.h>
#include <stdlib.h>

#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif
/*

Pour compiler
g++-9  -o binomial  binomial.c -fopenmp -O3

./binomial 10

1
1 1
1 2 1
1 3 3 1
1 4 6 4 1
1 5 10 10 5 1
1 6 15 20 15 6 1
1 7 21 35 35 21 7 1
1 8 28 56 70 56 28 8 1
1 9 36 84 126 126 84 36 9 1
*/

int main(int argc, char **argv) {
  constexpr int NMAX = 30;
  int array[NMAX][NMAX];

  int n;
  if (argc > 1) {
    n = atoi(argv[1]);
  } else {
    n = 20;
  }

  if (n > 30) {
    std::cerr << "N too large: keep under " << NMAX << std::endl;
    std::exit(EXIT_FAILURE);
  }

#pragma omp parallel
#pragma omp single
  for (int row = 1; row <= n; row++) {
    array[row][row] = 1;
    array[row][1] = 1;
    for (int col = 2; col < row; col++) {
      #pragma omp task firstprivate(row, col) shared(array)
      {
        array[row][col] = array[row - 1][col - 1] + array[row - 1][col];
      }
    }
  }

  for (int row = 1; row <= n; row++) {
    for (int col = 1; col <= row; col++) {
      int binom = array[row][col];
      std::cout << binom << " ";
    }
    std::cout << "\n";
  }
  return 0;
}
