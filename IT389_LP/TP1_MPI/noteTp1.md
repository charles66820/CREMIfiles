# TP1 MPI

## Exercice 0

`mpicc -show` me donne :

> Sur mon pc

```txt
gcc -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include -pthread -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi
```

> Sur PLAFrim `module load mpi/openmpi/3.1.4-all`

```txt
gcc -I/cm/shared/modules/generic/mpi/openmpi/3.1.4-all/include -pthread -Wl,-rpath -Wl,/cm/shared/modules/generic/mpi/openmpi/3.1.4-all/lib -Wl,--enable-new-dtags -L/cm/shared/modules/generic/mpi/openmpi/3.1.4-all/lib -lmpi
```

## Exercice 1

> `mpicc rankSize.c -o rankSize`

```bash
mpirun rankSize
Rank 2 pour 4 nodes
Rank 3 pour 4 nodes
Rank 0 pour 4 nodes
Rank 1 pour 4 nodes
```

## Exercice 2

```bash
mpirun rankSize
Rank pairs 1 pour 4 nodes
Rank impairs 2 pour 4 nodes
Rank impairs 0 pour 4 nodes
Rank pairs 3 pour 4 nodes
```

## Exercice 3

> mpicc simpleCom.c -o simpleCom

```bash
mpirun simpleCom
P0 send : {1.000000, 8.600000, -9.200000, 1.200000, 2.100000, 200.800003, 7.990000, 95.449997, 787.400024, -6.000000}
P1 receive (src=0, tag=100, err=0) : {1.000000, 8.600000, -9.200000, 1.200000, 2.100000, 200.800003, 7.990000, 95.449997, 787.400024, -6.000000}
```

## Exercice 4

```bash
mpirun simpleCom
P0 send : {1.000000, 8.600000, -9.200000, 1.200000, 2.100000, 200.800003, 7.990000, 95.449997, 787.400024, -6.000000}
P1 receive (src=0, tag=100, err=0) : {1.000000, 8.600000, -9.200000, 1.200000, 2.100000, 200.800003, 7.990000, 95.449997, 787.400024, -6.000000}
P1 send : {-9.000000, -1.400000, -19.200001, -8.800000, -7.900000, 190.800003, -2.010000, 85.449997, 777.400024, -16.000000}
P0 receive (src=1, tag=101, err=0) : {-9.000000, -1.400000, -19.200001, -8.800000, -7.900000, 190.800003, -2.010000, 85.449997, 777.400024, -16.000000}
```

## Exercice 5

```bash
mpirun valueForwarding
P0 send : 0.000000
P1 receive (src=0, tag=100, err=0) : 0.000000
P1 send : 0.000000
P2 receive (src=1, tag=100, err=0) : 0.000000
P2 send : 0.000000
P3 receive (src=2, tag=100, err=0) : 0.000000
```

## Exercice 6


