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

  int rank, N;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &N);

  srand((unsigned int)time(NULL) + rank);

  size_t size = 10;
  int contactRank = 0;
  type tab[size];
  type rTab[size];
  MPI_Status statusSend;
  MPI_Status statusRecv;
  MPI_Request reqSend;
  MPI_Request reqRecv;

  if (rank == 0 || rank == 1) {
    if (rank == 0) {
      for (size_t i = 0; i < size; i++) tab[i] = RAND_TYPE(40);
      contactRank = 1;
    } else if (rank == 1) {
      for (size_t i = 0; i < size; i++) tab[i] = RAND_TYPE(40);
      contactRank = 0;
    }
    printf("P%d send : ", rank);
    printTab(tab, size);
    MPI_Isend(&tab, size, MPI_FLOAT, contactRank, 100, MPI_COMM_WORLD,
              &reqSend);
    MPI_Irecv(&rTab, size, MPI_FLOAT, contactRank, 100, MPI_COMM_WORLD,
              &reqRecv);
    MPI_Wait(&reqSend, &statusSend);
    MPI_Wait(&reqRecv, &statusRecv);
    printf("P%d receive : ", rank);
    printTab(rTab, size);
  }

  MPI_Finalize();

  return 0;
}
