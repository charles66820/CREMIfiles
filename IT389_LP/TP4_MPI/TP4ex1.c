#include <stdio.h>
#include <math.h>
#include "mpi.h"


int main(int argc, char** argv){

	int myrank, nprocs; 
	int my_value = 12345;

  MPI_Init(&argc,&argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	int window_buffer[100] = {0};
	MPI_Win win;
	MPI_Win_create(&window_buffer, 100*sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);


	MPI_Win_fence(0,win);

	if(myrank == 0 || myrank == 2){
		MPI_Put(&my_value, 1, MPI_INT, 1, 80, 1, MPI_INT, win);
	}

	MPI_Win_fence(0,win);

	MPI_Win_free(&win);
	printf("I passed the win_free  !\n");


  MPI_Finalize();
  return 0;
}
