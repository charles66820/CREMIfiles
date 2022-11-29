#include <float.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

void userSum(int *r, int *l, int *len, MPI_Datatype *dType) { *r += *l; }

int main(int argc, char *argv[]) {
  srand((unsigned int)time(NULL));

  MPI_Init(&argc, &argv);

  int rank, N;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &N);

  MPI_Op sumOp;
  MPI_Op_create((MPI_User_function *)userSum, 1, &sumOp);

  int sum = 0;
  MPI_Allreduce(&rank, &sum, 1, MPI_INT, sumOp, MPI_COMM_WORLD);
  printf("P%d receive the sum : %d\n", rank, sum);

  MPI_Op_free(&sumOp);
  MPI_Finalize();

  return 0;
}
