#include <stdio.h>
#include <math.h>

int main() {
    int devID = 0;
    cudaDeviceProp deviceProps;
    cudaError_t err = cudaGetDeviceProperties(&deviceProps, devID);
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    int maxThreadsPerBlock = deviceProps.maxThreadsPerBlock;
    int numberOfSMs = deviceProps.multiProcessorCount;
    int maxThreadsPerMultiProcessor = deviceProps.maxThreadsPerMultiProcessor;
    int maxBlocksPerMultiProcessor = deviceProps.maxBlocksPerMultiProcessor;
    int warpSize = deviceProps.warpSize;
    
    printf("GPU Name: %s\n", deviceProps.name);
    int memorysize = deviceProps.totalGlobalMem / pow(1024, 3);
    printf("Total Global Memory: %d GB\n", memorysize);
    printf("Number of SMs: %d\n", numberOfSMs);
    printf("Max threads per multiprocessor: %d\n", maxThreadsPerMultiProcessor);
    printf("Max threads per block: %d\n", maxThreadsPerBlock);
    printf("Max blocks per multiprocessor: %d\n", maxBlocksPerMultiProcessor);
    printf("Warp size: %d\n", warpSize);

    return 0;
}
