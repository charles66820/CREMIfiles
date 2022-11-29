#include <float.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

int main(int argc, char* argv[]) {
  srand((unsigned int)time(NULL));

  MPI_Init(&argc, &argv);

  int rank, N;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &N);

  int sum = 0;

  if (rank == 0) {
    for (size_t i = 0; i < N; i++) sum += i;
    printf("P%d send local sum : %d\n", rank, sum);
  }

  MPI_Bcast(&sum, 1, MPI_INT, 0, MPI_COMM_WORLD);
  printf("P%d receive the sum : %d\n", rank, sum);

  MPI_Finalize();

  return 0;
}
