/* Fichier deviceQuery.cu (Sept 2015) */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <cuda_runtime.h>
#include <helper_cuda.h> // Dans /usr/local/cuda/samples/common/inc/ pour checkCudaErrors
// Fonction pour le calcul du nombre de cores
int ConvertSMVer2Cores(int major, int minor)
{
	// Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
	typedef struct {
		int SM; // 0xMm (hexadecimal notation), M = SM Major version, and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] = 
	{ { 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
	  { 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
	  { 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
	  { 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
	  { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
	  { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
	  { 0x30, 192}, // Fermi Generation (SM 3.0) GK10x class
	  {   -1, -1 }
	};

	int index = 0;
	while (nGpuArchCoresPerSM[index].SM != -1) {
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor) ) {
			return nGpuArchCoresPerSM[index].Cores;
		}
		index++;
	}
	printf("MapSMtoCores SM %d.%d is undefined (please update to the latest SDK)!\n", major, minor);
	return -1;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
    int deviceCount;
    int driverVersion=0; int runtimeVersion=0;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));
    //cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0)
        printf("There is no device supporting CUDA\n");
    int dev;
    for (dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
        if (dev == 0) {
            if (deviceProp.major == 9999 && deviceProp.minor == 9999)
                printf("There is no device supporting CUDA.\n");
            else if (deviceCount == 1)
                printf("There is 1 device supporting CUDA\n");
            else
                printf("There are %d devices supporting CUDA\n", deviceCount);
        }
        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);
        printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion/1000,
               (driverVersion%100)/10,runtimeVersion/1000, (runtimeVersion%100)/10);
        printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
               deviceProp.major, deviceProp.minor);
        printf("  Total amount of global memory:                 %u bytes\n",
               deviceProp.totalGlobalMem);
	printf("  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n",
               deviceProp.multiProcessorCount,
               _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
               _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
        printf("  Total amount of constant memory:               %u bytes\n",
               deviceProp.totalConstMem); 
        printf("  Total amount of shared memory per block:       %u bytes\n",
               deviceProp.sharedMemPerBlock);
        printf("  Total number of registers available per block: %d\n",
               deviceProp.regsPerBlock);
        printf("  Warp size:                                     %d\n",
               deviceProp.warpSize);
        printf("  Maximum number of threads per block:           %d\n",
               deviceProp.maxThreadsPerBlock);
        printf("  Maximum number of threads per multiprocessor:  %d\n",
               deviceProp.maxThreadsPerMultiProcessor);
        printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
               deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);
        printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
               deviceProp.maxGridSize[0],
               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
        printf("  Maximum memory pitch:                          %u bytes\n",
               deviceProp.memPitch);
        printf("  Texture alignment:                             %u bytes\n",
               deviceProp.textureAlignment);
        printf("  Clock rate:                                    %.2f GHz\n",
               deviceProp.clockRate * 1e-6f);
        printf("  Memory clock rate:                             %.2f GHz\n",
               deviceProp.memoryClockRate * 1e-6f);
        printf("  Memory Bus Width:                              %.d bits\n",
               deviceProp.memoryBusWidth );
    #if CUDART_VERSION >= 2000
        printf("  Concurrent copy and execution:                 %s\n",
               deviceProp.deviceOverlap ? "Yes" : "No");
    #endif
    }
    printf("\nTest PASSED\n");
}
