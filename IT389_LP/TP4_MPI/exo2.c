#include <float.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define SIZE 10

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int rank, N;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &N);

  MPI_Win win;

#if 0
  int sharedBuf[SIZE] = {0};
  MPI_Win_create(&sharedBuf, SIZE * sizeof(int), sizeof(int), MPI_INFO_NULL,
                 MPI_COMM_WORLD, &win);
#else
  int* sharedBuf;
  MPI_Win_allocate(SIZE * sizeof(int), sizeof(int), MPI_INFO_NULL,
                   MPI_COMM_WORLD, &sharedBuf, &win);
#endif

  int value = 12345;


  if (rank >= 0) {
    int buffPos = 6;// disp

    MPI_Win_lock(0, win);

    MPI_Put(&value, 1 /* nb */, MPI_INT, 1 /* memory rank target */, buffPos, 1 /* nb */, MPI_INT, win);
    // TODO: Post-Start-Complete-Wait,

    MPI_Win_unlock(win);
  }

  printf("P%d value after put : %d\n", rank, sharedBuf[buffPos]);

  MPI_Win_free(&win);

  MPI_Finalize();

  return 0;
}
