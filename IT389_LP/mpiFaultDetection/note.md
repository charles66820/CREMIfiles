# MPI Faults detection

## Use parcoach

```bash
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank, N;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &N);

  if (rank % 2)
    MPI_Barrier(MPI_COMM_WORLD);

  MPI_Finalize();
}
```

```bash
module load tools/parcoach/2.0.0 mpi/openmpi/4.1.1
clang -g -S -O1 -emit-llvm test.c -o test.s
cat test.s
parcoach --check-mpi --disable-output test.s
opt -dot-cfg test.s
dot -Tpng .main.dot -o main.png
open main.png
```
