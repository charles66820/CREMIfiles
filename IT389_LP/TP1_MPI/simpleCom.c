#include <float.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

void pingPongExo4() {
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
    MPI_Send(&tab, size, MPI_FLOAT, 1, 100, MPI_COMM_WORLD);

    MPI_Recv(&tab, size, MPI_FLOAT, 1, 101, MPI_COMM_WORLD, &status);
    printf("P%d receive (src=%d, tag=%d, err=%d) : ", rank, status.MPI_SOURCE,
           status.MPI_TAG, status.MPI_ERROR);
    printTab(tab, size);
  } else if (rank == 1) {
    type tab[size];

    MPI_Recv(&tab, size, MPI_FLOAT, 0, 100, MPI_COMM_WORLD, &status);
    printf("P%d receive (src=%d, tag=%d, err=%d) : ", rank, status.MPI_SOURCE,
           status.MPI_TAG, status.MPI_ERROR);
    printTab(tab, size);

    for (size_t i = 0; i < size; i++) tab[i] -= 10;

    printf("P%d send : ", rank);
    printTab(tab, size);
    MPI_Send(&tab, size, MPI_FLOAT, 0, 101, MPI_COMM_WORLD);
  }
}

void pingPongTimeExo6() {
  int rank, N;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &N);

  size_t size = 10;
  MPI_Status status;

  double timeStart = 0;
  double timeSum = 0;

  int count = 4000;

  if (rank == 0) printf("rank, time, size\n");

  for (size_t i = 0; i < count; i++) {
    if (rank == 0) {
      type tab[size];
      for (size_t i = 0; i < size; i++) tab[i] = RAND_TYPE(40);

      timeStart = MPI_Wtime();
      MPI_Send(&tab, size, MPI_FLOAT, 1, 100, MPI_COMM_WORLD);
      MPI_Recv(&tab, size, MPI_FLOAT, 1, 101, MPI_COMM_WORLD, &status);
      timeSum += (MPI_Wtime() - timeStart);

      printf("%d, %f, %ld\n", rank, timeSum, size);
    } else if (rank == 1) {
      type tab[size];

      timeStart = MPI_Wtime();
      MPI_Recv(&tab, size, MPI_FLOAT, 0, 100, MPI_COMM_WORLD, &status);
      timeSum += (MPI_Wtime() - timeStart);

      for (size_t i = 0; i < size; i++) tab[i] -= 10;

      timeStart = MPI_Wtime();
      MPI_Send(&tab, size, MPI_FLOAT, 0, 101, MPI_COMM_WORLD);
      timeSum += (MPI_Wtime() - timeStart);

      printf("%d, %f, %ld\n", rank, timeSum, size);
    }
    size += 10;
  }
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  srand((unsigned int)time(NULL));

  // pingPongExo4();
  pingPongTimeExo6();

  MPI_Finalize();

  return 0;
}
