CC      = gcc
CFLAGS += -Wall -g -O4
LDLIBS += -lm -lrt

all: stencil_seq stencil_MPI_Pure stencil_OMP

clean:
	-rm stencil_seq stencil_OMP stencil_MPI_Pure *.out

diff: all
	./stencil_seq > stencil_seq.out
	./stencil_OMP > stencil_OMP.out
	./stencil_MPI_Pure > stencil_MPI_Pure.out
	diff stencil_seq.out stencil_OMP.out
	diff stencil_seq.out stencil_MPI_Pure.out

mrproper: clean
	-rm *~

archive: stencil.c Makefile
	( cd .. ; tar czf stencil.tar.gz stencil/ )

