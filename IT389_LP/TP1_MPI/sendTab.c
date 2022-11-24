#include <float.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define type int

#define RAND_TYPE(maxVal) (type) rand() / (type)(RAND_MAX / maxVal)

void printTab(type tab[], size_t size) {
  printf("{");
  for (size_t i = 0; i < size; i++) {
    if (i != 0) printf(", ");
    printf("%d", tab[i]);
  }
  printf("}\n");
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  srand((unsigned int)time(NULL));

 int rank, N;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &N);

  size_t size = 10;
  MPI_Status status;

  if (rank == 0) {
    type tab[size];
    for (size_t i = 0; i < size; i++) tab[i] = RAND_TYPE(40);

    printf("P%d send : ", rank);
    printTab(tab, size);
    for (size_t i = 1; i < N; i++)
      MPI_Send(&tab, size, MPI_FLOAT, i, 100, MPI_COMM_WORLD);
  } else {
    type tab[size];

    MPI_Recv(&tab, size, MPI_FLOAT, 0, 100, MPI_COMM_WORLD, &status);
    printf("P%d receive (src=%d, tag=%d, err=%d) : ", rank, status.MPI_SOURCE,
           status.MPI_TAG, status.MPI_ERROR);
    printTab(tab, size);
  }

  MPI_Finalize();

  return 0;
}
