#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank, N;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &N);

  int val, valRecv;
  MPI_Status statusSend;
  MPI_Status statusRecv;
  MPI_Request reqSend;
  MPI_Request reqRecv;

  if (rank == 0) {
    val = 1;
    printf("P%d send : %d\n", rank, val);
    MPI_Isend(&val, 1, MPI_DOUBLE, 1, 100, MPI_COMM_WORLD, &reqSend);
    MPI_Irecv(&valRecv, 1, MPI_DOUBLE, N-1, 100, MPI_COMM_WORLD, &reqRecv);

    MPI_Wait(&reqSend, &statusSend);
    MPI_Wait(&reqRecv, &statusRecv);
    printf("P%d receive (src=%d, tag=%d, err=%d) : %d\n", rank, statusRecv.MPI_SOURCE, statusRecv.MPI_TAG, statusRecv.MPI_ERROR, valRecv);
  } else {
    MPI_Irecv(&valRecv, 1, MPI_DOUBLE, rank-1, 100, MPI_COMM_WORLD, &reqRecv);
    MPI_Wait(&reqRecv, &statusRecv);
    printf("P%d receive (src=%d, tag=%d, err=%d) : %d\n", rank, statusRecv.MPI_SOURCE, statusRecv.MPI_TAG, statusRecv.MPI_ERROR, valRecv);
    valRecv += 1;
    if (rank + 1 < N) {
      printf("P%d send : %d\n", rank, valRecv);
      MPI_Isend(&valRecv, 1, MPI_DOUBLE, rank-1, 100, MPI_COMM_WORLD, &reqSend);
      MPI_Wait(&reqSend, &statusSend);
    } else if (rank == N-1) {
      printf("P%d send : %d\n", rank, valRecv);
      MPI_Isend(&valRecv, 1, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD, &reqSend);
      MPI_Wait(&reqSend, &statusSend);
    }
  }

  MPI_Finalize();

  return 0;
}
