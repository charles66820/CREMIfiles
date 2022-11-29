#include <float.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
  srand((unsigned int)time(NULL));

  MPI_Init(&argc, &argv);

  int rank, N;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &N);

  MPI_Comm newComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &newComm);

  MPI_Comm newSplitComm;
  MPI_Comm_split(newComm, rank % 2, rank, &newSplitComm);

  int newSplitCommRank, newSplitCommN;
  MPI_Comm_rank(newSplitComm, &newSplitCommRank);
  MPI_Comm_size(newSplitComm, &newSplitCommN);

  printf("rank = %d/%d, newSplitComm = %d/%d\n", rank, N, newSplitCommRank, newSplitCommN);

  MPI_Comm_free(&newComm);
  MPI_Comm_free(&newSplitComm);
  MPI_Finalize();

  return 0;
}
