#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank, N;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &N);

  if (rank % 2 == 0)
    printf("Rank impairs %d pour %d nodes\n", rank, N);
  else
    printf("Rank pairs %d pour %d nodes\n", rank, N);

  MPI_Finalize();

  return 0;
}
