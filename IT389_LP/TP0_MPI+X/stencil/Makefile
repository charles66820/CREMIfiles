CC      = gcc
CFLAGS += -Wall -g -O4
LDLIBS += -lm -lrt

all: stencil_seq stencil_MPI_Pure

clean:
	-rm stencil_seq stencil_MPI_Pure

mrproper: clean
	-rm *~

archive: stencil.c Makefile
	( cd .. ; tar czf stencil.tar.gz stencil/ )

