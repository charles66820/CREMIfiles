/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <assert.h>

#include "mpi.h"

int main(int argc, char *argv[])
{
    int rank;
    int size;

    /* initialiser MPI, à faire _toujours_ en tout premier du programme */
    MPI_Init(&argc, &argv);

    /* récupérer son propre numéro */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* récupérer le nombre de processus lancés */
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    printf("Hello world from process %d of %d from %s\n", rank, size, processor_name);

    assert(size % 2 == 0);
    MPI_Status status;

    if (rank % 2 == 1)
    {
        char buf[1];
        MPI_Recv(buf, 1, MPI_CHAR, rank - 1, 0, MPI_COMM_WORLD, &status);
        printf("%d recive (%c) from %d\n", rank, buf[0], rank - 1);
    }
    else
    {
        char buf[1] = {'B'};
        MPI_Send(buf, 1, MPI_CHAR, rank + 1, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
