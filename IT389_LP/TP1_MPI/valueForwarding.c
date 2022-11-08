#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank, N;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &N);

  MPI_Status status;
  double val;

  if (rank == 0) {
    val = MPI_Wtime();
    printf("P%d send : %f\n", rank, val);
    MPI_Send(&val, 1, MPI_DOUBLE, 1, 100, MPI_COMM_WORLD);
  } else {
    MPI_Recv(&val, 1, MPI_DOUBLE, rank-1, 100, MPI_COMM_WORLD, &status);
    printf("P%d receive (src=%d, tag=%d, err=%d) : %f\n", rank, status.MPI_SOURCE, status.MPI_TAG, status.MPI_ERROR, val);
    if (rank + 1 < N) {
      printf("P%d send : %f\n", rank, val);
      MPI_Send(&val, 1, MPI_DOUBLE, rank+1, 100, MPI_COMM_WORLD);
    }
  }

  MPI_Finalize();

  return 0;
}
