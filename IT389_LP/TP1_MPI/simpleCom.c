#include <mpi.h>
#include <stdio.h>

#define type float

void printTab(type tab[], size_t size) {
  printf("{");
  for (size_t i = 0; i < size; i++) {
    if (i != 0) printf(", ");
    printf("%f", tab[i]);
  }
  printf("}\n");
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank, N;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &N);

  size_t size = 10;
  MPI_Status status;

  if (rank == 0) {
    type tab[] = {1.0, 8.6, -9.2, 1.2, 2.1, 200.8, 7.99, 95.45, 787.4, -6.0};
    printf("P%d send : ", rank);
    printTab(tab, size);
    MPI_Send(&tab, size, MPI_FLOAT, 1, 100, MPI_COMM_WORLD);

    MPI_Recv(&tab, size, MPI_FLOAT, 1, 101, MPI_COMM_WORLD, &status);
    printf("P%d receive (src=%d, tag=%d, err=%d) : ", rank, status.MPI_SOURCE, status.MPI_TAG, status.MPI_ERROR);
    printTab(tab, size);
  } else if (rank == 1) {
    type tab[size];
    MPI_Recv(&tab, size, MPI_FLOAT, 0, 100, MPI_COMM_WORLD, &status);
    printf("P%d receive (src=%d, tag=%d, err=%d) : ", rank, status.MPI_SOURCE, status.MPI_TAG, status.MPI_ERROR);
    printTab(tab, size);

    for (size_t i = 0; i < size; i++)
      tab[i] -= 10;

    printf("P%d send : ", rank);
    printTab(tab, size);
    MPI_Send(&tab, size, MPI_FLOAT, 0, 101, MPI_COMM_WORLD);
  }

  MPI_Finalize();

  return 0;
}
