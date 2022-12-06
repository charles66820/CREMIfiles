#include <float.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define SIZE 2

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int rank, N;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &N);

  if (rank >= 0 && rank <= 1) {
    MPI_Win win;
    int sendBuf = 0, recvBuf = 0;

#if 1
    int sharedBuf[SIZE] = {0};
    MPI_Win_create(&sharedBuf, SIZE * sizeof(int), sizeof(int), MPI_INFO_NULL,
                   MPI_COMM_WORLD, &win);
#else
    int* sharedBuf;
    MPI_Win_allocate(SIZE * sizeof(int), sizeof(int), MPI_INFO_NULL,
                     MPI_COMM_WORLD, &sharedBuf, &win);
#endif

    TODO: fix
    sendBuf = 12345;

    printf("P%d value before put : %d\n", rank, sendBuf);
    // MPI_Put(&sendBuf, 1, MPI_INT, rank == 0 ? 1 : 0, sizeof(int), 1, MPI_INT,
    //         win);
    printf("P%d value after put : %d\n", rank, recvBuf);

    // MPI_Win_free(&win);
  }

  MPI_Finalize();

  return 0;
}
