CFLAGS += -g -D_REENTRANT
LDLIBS = -lpthread # -pthread

distribut: distribut.o distributor.o
distribut.o: distribut.c distributor.h
distributor.o: distributor.h distributor.c

clean:
	rm *.o distribut
