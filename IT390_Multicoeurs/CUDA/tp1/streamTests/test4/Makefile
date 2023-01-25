CC=nvcc

CUDA_HOME=/net/ens/nvhpc/hpc_sdk/21.3/Linux_x86_64/21.3/cuda/
CUDASDK=/net/ens/nvhpc/hpc_sdk/21.3/Linux_x86_64/21.3/examples/OpenACC/SDK/include/
CUDANVIDIA=/usr/lib/nvidia
NVCC=nvcc

NVCFLAGS= -I $(CUDA_HOME)/include -I$(CUDASDK)

CFLAGS= -L $(CUDA_HOME)/lib64 -L $(CUDANVIDIA)

saxpy: saxpy_solution.o
	$(NVCC) $(CFLAGS)  -o $@ $^

saxpy_stream: saxpy_solution_streams.o
	$(NVCC) $(CFLAGS)  -o $@ $^

saxpy_stream_fix: saxpy_solution_streams_fix.o
	$(NVCC) $(CFLAGS)  -o $@ $^ 

clean:
	rm -f *.o *.linkinfo deviceQuery

%.o: %.cu
	$(NVCC) $(NVCFLAGS) -c -o $@ $<
