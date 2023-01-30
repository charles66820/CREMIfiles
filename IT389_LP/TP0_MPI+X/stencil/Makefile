CC      = gcc
CFLAGS += -Wall -g -O4
LDLIBS += -lm -lrt

all: stencil_seq stencil_halos stencil_OMP_for stencil_OMP_task stencil_MPI_Pure

clean:
	-rm -f stencil_seq stencil_halos stencil_OMP_for stencil_OMP_task stencil_MPI_Pure *.out

diff: all
	./stencil_seq > stencil_seq.out
	./stencil_halos > stencil_halos.out
	./stencil_OMP_for > stencil_OMP_for.out
	./stencil_OMP_task > stencil_OMP_task.out
	./stencil_MPI_Pure > stencil_MPI_Pure.out
	diff stencil_seq.out stencil_halos.out
	diff stencil_seq.out stencil_OMP_for.out
	diff stencil_seq.out stencil_OMP_task.out
	diff stencil_seq.out stencil_MPI_Pure.out

mrproper: clean
	-rm *~

archive: stencil.c Makefile
	( cd .. ; tar czf stencil.tar.gz stencil/ )

