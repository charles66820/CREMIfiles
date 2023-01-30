STENCIL_SIZE_X ?= 25
STENCIL_SIZE_Y ?= 30
TILE_WIDTH ?= 6
TILE_HEIGHT ?= 6

CC      = gcc
CFLAGS += -DSTENCIL_SIZE_X=$(STENCIL_SIZE_X) -DSTENCIL_SIZE_Y=$(STENCIL_SIZE_Y)
CFLAGS += -DTILE_WIDTH=$(TILE_WIDTH) -DTILE_HEIGHT=$(TILE_HEIGHT)
# CFLAGS += -Wall -g -O4
CFLAGS += -fopenmp
LDLIBS += -lm -lrt

all: stencil_seq stencil_halos stencil_OMP_for stencil_OMP_task stencil_MPI_Pure

clean:
	-rm -f stencil_seq stencil_halos stencil_OMP_for stencil_OMP_task stencil_MPI_Pure *.out

diff: all
	./stencil_seq -d > stencil_seq.out
	./stencil_halos -d > stencil_halos.out
	./stencil_OMP_for -d > stencil_OMP_for.out
	./stencil_OMP_task -d > stencil_OMP_task.out
	./stencil_MPI_Pure -d > stencil_MPI_Pure.out
	diff stencil_seq.out stencil_halos.out
	diff stencil_seq.out stencil_OMP_for.out
	diff stencil_seq.out stencil_OMP_task.out
	diff stencil_seq.out stencil_MPI_Pure.out

mrproper: clean
	-rm *~

archive: stencil.c Makefile
	( cd .. ; tar czf stencil.tar.gz stencil/ )

