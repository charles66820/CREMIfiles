CC=gcc

CUDA_HOME=/net/ens/nvhpc/hpc_sdk/21.3/Linux_x86_64/21.3/cuda/
CUDASDK=/net/ens/nvhpc/hpc_sdk/21.3/Linux_x86_64/21.3/examples/OpenACC/SDK/include/
CUDANVIDIA=/usr/lib/nvidia
NVCC=$(CUDA_HOME)/bin/nvcc

NVCFLAGS= -I $(CUDA_HOME)/include -I$(CUDASDK)

CFLAGS= -L $(CUDA_HOME)/lib64 -L $(CUDANVIDIA) -lGLU -lGL  -lglut

vague: vague.o
	$(NVCC) $(CFLAGS)  -o $@ $^

clean:
	rm -f *.o *.linkinfo vague

%.o: %.cu
	$(NVCC) $(NVCFLAGS) -c -o $@ $<

