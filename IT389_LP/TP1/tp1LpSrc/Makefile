
all: seq ompSIMD

run: seqRun ompSIMDRun

seq:
	g++ -O3 -o mainSeq main.cpp

seqRun:
	./mainSeq > seq.out

ompSIMD:
	g++ -fopenmp -O3 -o mainOmp main.cpp

ompSIMDRun:
	./mainOmp > ompSIMD.out