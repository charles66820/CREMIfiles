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
  MPI_Init(&argc, &argv);

  srand((unsigned int)time(NULL));

  int rank, N;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &N);

  size_t size = 10;

  type tab[size];
  if (rank == 0) {
    for (size_t i = 0; i < size; i++) tab[i] = RAND_TYPE(40);
    printf("P%d send : ", rank);
    printTab(tab, size);
  } else {
    for (size_t i = 0; i < size; i++) tab[i] = 0;
    printf("P%d have before broadcast : ", rank);
    printTab(tab, size);
  }

  MPI_Bcast(tab, size, MPI_INT, 0, MPI_COMM_WORLD);
  printf("P%d have after broadcast : ", rank);
  printTab(tab, size);

  MPI_Finalize();

  return 0;
}
