#include <float.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define type float

#define RAND_TYPE(maxVal) (type) rand() / (type)(RAND_MAX / maxVal)

void printTab(type tab[], size_t size) {
  printf("{");
  for (size_t i = 0; i < size; i++) {
    if (i != 0) printf(", ");
    printf("%f", tab[i]);
  }
  printf("}\n");
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int rank, N;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &N);

  type locFloat;
  type receiveFloats[N];

  srand((unsigned int)time(NULL) + rank);
  locFloat = RAND_TYPE(40);
  printf("P%d have local float : %f\n", rank, locFloat);

  MPI_Gather(&locFloat, 1, MPI_FLOAT, &receiveFloats, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    printf("P%d receive this floats :\n", rank);
    printTab(receiveFloats, N);
  }

  MPI_Finalize();

  return 0;
}
