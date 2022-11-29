#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define type int

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank, N;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &N);

  size_t size = 10;
  MPI_Status status;

  for (size = 0; size < 2000; size++)
    if (rank == 0) {
      type tab[size];
      for (size_t i = 0; i < size; i++) tab[i] = 0;

      MPI_Send(&tab, size, MPI_FLOAT, 1, 100, MPI_COMM_WORLD);
      printf("Exec with size %ld\n", size);
    } else if (rank == 1) {
      type tab[size];
      // MPI_Recv(&tab, size, MPI_FLOAT, 0, 100, MPI_COMM_WORLD, &status);
    }

  MPI_Finalize();

  return 0;
}
