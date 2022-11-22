#include <float.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define type int

#define RAND_TYPE(maxVal) (type) rand() / (type)(RAND_MAX / maxVal)

void printTab(type tab[], size_t size) {
  printf("{");
  for (size_t i = 0; i < size; i++) {
    if (i != 0) printf(", ");
    printf("%d", tab[i]);
  }
  printf("}\n");
}

int main(int argc, char* argv[]) {
  srand((unsigned int)time(NULL));

  size_t size = 42;
  type tab[size];
  for (size_t i = 0; i < size; i++) tab[i] = RAND_TYPE(40);

  MPI_Init(&argc, &argv);

  int rank, N;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &N);

  // Print
  if (rank == 0) {
    printf("The shared table : ");
    printTab(tab, size);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // Calc
  int partSize = size / N;
  int reminder = size % N;

  int locSum = 0, sum = 0;
  for (size_t i = partSize * rank;
       i < partSize * rank + partSize + (reminder * (rank == N - 1)); i++)
    locSum += tab[i];

  MPI_Reduce(&locSum, &sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  printf("The local sum is %d and total sum is %d for P%d\n", locSum, sum, rank);

  MPI_Finalize();

  return 0;
}
