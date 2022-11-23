CC=nvcc

CUDA_HOME=/net/ens/nvhpc/hpc_sdk/21.3/Linux_x86_64/21.3/cuda/
CUDASDK=/net/ens/nvhpc/hpc_sdk/21.3/Linux_x86_64/21.3/examples/OpenACC/SDK/include/
CUDANVIDIA=/usr/lib/nvidia
NVCC=nvcc

NVCFLAGS= -I $(CUDA_HOME)/include -I$(CUDASDK)

CFLAGS= -L $(CUDA_HOME)/lib64 -L $(CUDANVIDIA)

OBJS := ${SRCS:.c=.o}
OBJS_CUDA := ${SRCS:.cu=.o}

gemm_omp: matmul_openMP.o
	$(CC) -o $@ $^ -fopenmp

gemm: matmul.o
	$(NVCC) $(CFLAGS)  -o $@ $^

gemm_tiled: matmul_tiled.o
	$(NVCC) $(CFLAGS)  -o $@ $^

gemm_cublas: matmul_cublas.o
	$(NVCC) $(CFLAGS) -lcublas -o $@ $^

clean:
	rm -f *.o *.linkinfo matmul

%.o: %.cu
	$(NVCC) $(NVCFLAGS) -c -o $@ $<
