# TP2 MPI

## Exercice 1

```bash
mpirun exo1
P1 sleep for 3
P2 sleep for 1
P0 sleep for 2
P3 sleep for 2
P2 has wake-up with a coffee cup
P0 has wake-up with a coffee cup
P3 has wake-up with a coffee cup
P1 has wake-up with a coffee cup
P0 is sync
P3 is sync
P2 is sync
P1 is sync
```

## Exercice 2

```bash
mpirun exo2
P3 have before broadcast : {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
P0 send : {31, 37, 19, 8, 34, 23, 3, 23, 26, 39}
P0 have after broadcast : {31, 37, 19, 8, 34, 23, 3, 23, 26, 39}
P1 have before broadcast : {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
P2 have before broadcast : {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
P1 have after broadcast : {31, 37, 19, 8, 34, 23, 3, 23, 26, 39}
P3 have after broadcast : {31, 37, 19, 8, 34, 23, 3, 23, 26, 39}
P2 have after broadcast : {31, 37, 19, 8, 34, 23, 3, 23, 26, 39}
```

## Exercice 3

```bash
mpirun exo3
The shared table : {32, 37, 13, 31, 33, 13, 18, 15, 8, 31, 9, 2, 18, 23, 16, 37, 19, 21, 37, 3, 37, 31, 10, 37, 11, 38, 10, 23, 36, 11, 30, 28, 8, 4, 20, 2, 18, 38, 17, 26, 30, 27}
The local sum is 185 and total sum is 0 for P1
The local sum is 244 and total sum is 0 for P2
The local sum is 231 and total sum is 908 for P0
The local sum is 248 and total sum is 0 for P3
```

## Exercice 4

```bash
mpirun exo4
P0 have local float : 21.787653
P2 have local float : 10.586378
P3 have local float : 5.057123
P1 have local float : 36.366070
P0 receive this floats :
{21.787653, 36.366070, 10.586378, 5.057123}
```

## Exercice 5

```bash
mpirun exo5
P3 send local sum : 6
P1 send local sum : 6
P1 receive the sum : 24
P0 send local sum : 6
P0 receive the sum : 24
P3 receive the sum : 24
P2 send local sum : 6
P2 receive the sum : 24
```

## Exercice 6

```bash

```

TODO:

## Exercice 7

TODO:
