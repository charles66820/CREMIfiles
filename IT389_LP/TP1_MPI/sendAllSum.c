#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank, N;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &N);

  size_t size = 10;
  MPI_Status status;

  int sum = 0;
  int sumR;
  for (size_t i = 0; i < N; i++) sum += i;

  for (size_t i = 0; i < N; i++) {
    if (rank != i) {
        printf("P%d to %ld send : %d\n", rank, i, sum);
        MPI_Send(&sum, size, MPI_FLOAT, i, 100, MPI_COMM_WORLD);
    }
  }

  for (size_t i = 0; i < N - 1; i++) {
    MPI_Recv(&sumR, size, MPI_FLOAT, MPI_ANY_SOURCE, 100, MPI_COMM_WORLD,
             &status);
    printf("P%d receive (src=%d, tag=%d, err=%d) : %d\n", rank,
           status.MPI_SOURCE, status.MPI_TAG, status.MPI_ERROR, sumR);
  }  // TODO:

  MPI_Finalize();

  return 0;
}
