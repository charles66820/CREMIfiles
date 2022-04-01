
#ifdef ENABLE_MPI
#include <mpi.h>

static int rank, size;

void mandel_init_mpi()
{
    easypap_check_mpi(); // check if MPI was correctly configured

    // todo ...

    mandel_init();
}

static int rankTop(int rank)
{
    return 0; //todo
}

static int rankSize(int rank)
{
    return 0; // todo
}

void mandel_refresh_img_mpi()
{
    // todo
}

//////////// MPI basic varianr
// Suggested cmdline:
// ./run -k mandel -v mpi -mpi "-np 4"  -d M

unsigned mandel_compute_mpi(unsigned nb_iter)
{
    for (unsigned it = 1; it <= nb_iter; it++)
    {
        do_tile(0, rankTop(rank), DIM, rankSize(rank), 0);
        zoom();
    }
    return 0;
}
#endif