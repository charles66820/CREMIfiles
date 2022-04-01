CC=mpicc
MPIRUN=mpirun 
MACHINES=mymachines
CFLAGS=-g -Wall -Og
SOURCE=$(wildcard *.c)
OBJ=$(SOURCE:.c=.o)
PROG?=hellow
NP?=2

all: $(PROG)

clean:
	rm -f $(OBJ) $(PROG)

run: $(PROG)
	$(MPIRUN)  --oversubscribe --mca btl tcp,self -hostfile $(MACHINES) --map-by node -np $(NP) ./$(PROG)

lrun: $(PROG)
	$(MPIRUN) --mca btl tcp,self -np $(NP) ./$(PROG)

