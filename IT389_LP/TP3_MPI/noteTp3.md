# TP3 MPI

## Exercice 1

> v1

```txt
P0 send local sum : 6
P0 receive the sum : 6
P2 receive the sum : 6
P1 receive the sum : 6
P3 receive the sum : 6
```

> v2

```txt
P3 receive the sum : 6
P0 receive the sum : 6
P1 receive the sum : 6
P2 receive the sum : 6
```

> v3

```bash
P3 receive the sum : 6
P0 receive the sum : 6
P1 receive the sum : 6
P2 receive the sum : 6
```

FIXME:

## Exercice 2

```bash
mpicc exo2.c -o exo2 && mpirun exo2
rank = 0/4, newSplitComm = 0/2
rank = 1/4, newSplitComm = 0/2
rank = 2/4, newSplitComm = 1/2
rank = 3/4, newSplitComm = 1/2
```

## Exercice 3

TODO:

## Exercice 4

TODO:
