CC=cc

CUDA_HOME=/net/ens/nvhpc/hpc_sdk/21.3/Linux_x86_64/21.3/cuda/
CUDASDK=/net/ens/nvhpc/hpc_sdk/21.3/Linux_x86_64/21.3/examples/OpenACC/SDK/include/
CUDANVIDIA=/usr/lib/nvidia
NVCC=nvcc

NVCFLAGS= -I $(CUDA_HOME)/include -I$(CUDASDK)

CFLAGS= -L $(CUDA_HOME)/lib64 -L $(CUDANVIDIA)

OBJS := ${SRCS:.c=.o}
OBJS_CUDA := ${SRCS:.cu=.o}

test1: test1.o
	$(NVCC) $(CFLAGS)  -o $@ $^

test2: test2.o
	$(NVCC) $(CFLAGS)  -o $@ $^

test3: test3.o
	$(NVCC) $(CFLAGS)  -o $@ $^


clean:
	rm -f *.o *.linkinfo matmul gemm_cublas gemm_omp matmul_tiled gemm_cublas_solution gemm_omp_solution matmul_tiled_solution

%.o: %.cu
	$(NVCC) $(NVCFLAGS) -c -o $@ $<
