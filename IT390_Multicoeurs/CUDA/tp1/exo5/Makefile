#CC=gcc

CUDA_HOME=/net/ens/nvhpc/hpc_sdk/21.3/Linux_x86_64/21.3/cuda/
CUDASDK=/net/ens/nvhpc/hpc_sdk/21.3/Linux_x86_64/21.3/examples/OpenACC/SDK/include/
CUDANVIDIA=/usr/lib/nvidia
NVCC=nvcc

NVCFLAGS= -I $(CUDA_HOME)/include -I$(CUDASDK)

CFLAGS= -L $(CUDA_HOME)/lib64 -L $(CUDANVIDIA)

inversion_no_shared: inversion_no_shared.o
	$(NVCC) $(CFLAGS)  -o $@ $^

inversion_shared: inversion_shared.o
	$(NVCC) $(CFLAGS)  -o $@ $^

clean:
	rm -f *.o *.linkinfo inversion_no_shared inversion_shared

%.o: %.cu
	$(NVCC) $(NVCFLAGS) -c -o $@ $<
