#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank, N;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &N);

  int val;
  MPI_Status status;

  if (rank == 0) {
    val = 1;
    printf("P%d send : %d\n", rank, val);
    MPI_Send(&val, 1, MPI_DOUBLE, 1, 100, MPI_COMM_WORLD);

    MPI_Recv(&val, 1, MPI_DOUBLE, N-1, 100, MPI_COMM_WORLD, &status);
    printf("P%d receive (src=%d, tag=%d, err=%d) : %d\n", rank, status.MPI_SOURCE, status.MPI_TAG, status.MPI_ERROR, val);
  } else {
    MPI_Recv(&val, 1, MPI_DOUBLE, rank-1, 100, MPI_COMM_WORLD, &status);
    printf("P%d receive (src=%d, tag=%d, err=%d) : %d\n", rank, status.MPI_SOURCE, status.MPI_TAG, status.MPI_ERROR, val);
    val += 1;
    if (rank + 1 < N) {
      printf("P%d send : %d\n", rank, val);
      MPI_Send(&val, 1, MPI_DOUBLE, rank+1, 100, MPI_COMM_WORLD);
    } else if (rank == N-1) {
      printf("P%d send : %d\n", rank, val);
      MPI_Send(&val, 1, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD);
    }
  }

  MPI_Finalize();

  return 0;
}
