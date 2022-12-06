#include <float.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define RAND_TYPE(minVal, maxVal) (rand() % (maxVal - minVal)) + minVal;

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int rank, N;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &N);

  srand((unsigned int)time(NULL) + rank);

  int sleepTime = RAND_TYPE(1, 4);
  printf("P%d sleep for %d\n", rank, sleepTime);
  sleep(sleepTime);

  printf("P%d has wake-up with a coffee cup\n", rank);
  MPI_Barrier(MPI_COMM_WORLD);
  printf("P%d is sync\n", rank);

  MPI_Finalize();

  return 0;
}
