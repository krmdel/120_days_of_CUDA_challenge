#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define INF 0x3f3f3f3f
#define BLOCK_SIZE 256

// Macro to check for CUDA errors
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if(code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if(abort) exit(code);
   }
}

__global__ void bfs_top_down_kernel(int V, const int *row_offsets, const int *col_indices,
                                      int current_level, int *distances, const int *frontier, int frontier_size,
                                      int *next_frontier, int *next_frontier_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < frontier_size) {
        int u = frontier[tid];
        int row_start = row_offsets[u];
        int row_end = row_offsets[u+1];
        for (int offset = row_start; offset < row_end; offset++) {
            int v = col_indices[offset];
            if (atomicCAS(&distances[v], INF, current_level + 1) == INF) {
                int pos = atomicAdd(next_frontier_size, 1);
                next_frontier[pos] = v;
            }
        }
    }
}

__global__ void bfs_bottom_up_kernel(int V, const int *row_offsets, const int *col_indices,
                                       int current_level, int *distances, int *changed) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if(u < V && distances[u] == INF) {
        int row_start = row_offsets[u];
        int row_end = row_offsets[u+1];
        for (int offset = row_start; offset < row_end; offset++) {
            int v = col_indices[offset];
            if(distances[v] == current_level) {
                distances[u] = current_level + 1;
                *changed = 1;
                break;
            }
        }
    }
}

__global__ void bfs_edge_centric_kernel(int E, const int *edge_src, const int *edge_dst,
                                          int current_level, int *distances, int *changed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < E) {
        int u = edge_src[tid];
        int v = edge_dst[tid];
        if(distances[u] == current_level && distances[v] == INF) {
            if(atomicCAS(&distances[v], INF, current_level + 1) == INF) {
                *changed = 1;
            }
        }
        if(distances[v] == current_level && distances[u] == INF) {
            if(atomicCAS(&distances[u], INF, current_level + 1) == INF) {
                *changed = 1;
            }
        }
    }
}

__global__ void count_frontier_kernel(int V, const int *distances, int level, int *frontier_count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < V) {
        if(distances[tid] == level + 1)
            atomicAdd(frontier_count, 1);
    }
}

void run_bfs_top_down(const int *h_row_offsets, const int *h_col_indices, int V, int E, int source) {

    // Create CUDA events for timing.
    cudaEvent_t start, copyHtoD, kernelStart, kernelEnd, copyDtoH, end;
    cudaEventCreate(&start);
    cudaEventCreate(&copyHtoD);
    cudaEventCreate(&kernelStart);
    cudaEventCreate(&kernelEnd);
    cudaEventCreate(&copyDtoH);
    cudaEventCreate(&end);

    cudaEventRecord(start, 0);

    // Allocate and copy graph data to device
    int *d_row_offsets, *d_col_indices;
    gpuErrchk( cudaMalloc((void**)&d_row_offsets, sizeof(int)*(V+1)) );
    gpuErrchk( cudaMalloc((void**)&d_col_indices, sizeof(int)*E) );
    gpuErrchk( cudaMemcpy(d_row_offsets, h_row_offsets, sizeof(int)*(V+1), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_col_indices, h_col_indices, sizeof(int)*E, cudaMemcpyHostToDevice) );

    // Allocate distances array on device
    int *d_distances;
    gpuErrchk( cudaMalloc((void**)&d_distances, sizeof(int)*V) );
    gpuErrchk( cudaMemset(d_distances, 0x3f, sizeof(int)*V) );
    int zero = 0;
    gpuErrchk( cudaMemcpy(&d_distances[source], &zero, sizeof(int), cudaMemcpyHostToDevice) );

    // Allocate frontier arrays on device
    int *d_frontier_current, *d_frontier_next;
    gpuErrchk( cudaMalloc((void**)&d_frontier_current, sizeof(int)*V) );
    gpuErrchk( cudaMalloc((void**)&d_frontier_next, sizeof(int)*V) );
    gpuErrchk( cudaMemcpy(d_frontier_current, &source, sizeof(int), cudaMemcpyHostToDevice) );
    int frontier_size = 1;

    int *d_frontier_size;
    gpuErrchk( cudaMalloc((void**)&d_frontier_size, sizeof(int)) );

    cudaEventRecord(copyHtoD, 0);

    // Begin BFS
    int level = 0;
    cudaEventRecord(kernelStart, 0);
    while(frontier_size > 0) {
        gpuErrchk( cudaMemset(d_frontier_size, 0, sizeof(int)) );

        int num_blocks = (frontier_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        bfs_top_down_kernel<<<num_blocks, BLOCK_SIZE>>>(V, d_row_offsets, d_col_indices,
                                                          level, d_distances, d_frontier_current, frontier_size,
                                                          d_frontier_next, d_frontier_size);
        gpuErrchk( cudaDeviceSynchronize() );

        int new_frontier_size;
        gpuErrchk( cudaMemcpy(&new_frontier_size, d_frontier_size, sizeof(int), cudaMemcpyDeviceToHost) );
        frontier_size = new_frontier_size;

        int *temp = d_frontier_current;
        d_frontier_current = d_frontier_next;
        d_frontier_next = temp;
        level++;
    }
    cudaEventRecord(kernelEnd, 0);

    int *h_distances = (int*)malloc(sizeof(int)*V);
    gpuErrchk( cudaMemcpy(h_distances, d_distances, sizeof(int)*V, cudaMemcpyDeviceToHost) );
    cudaEventRecord(copyDtoH, 0);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

    float time_copyHtoD, time_kernel, time_copyDtoH, time_total;
    cudaEventElapsedTime(&time_copyHtoD, start, copyHtoD);
    cudaEventElapsedTime(&time_kernel, kernelStart, kernelEnd);
    cudaEventElapsedTime(&time_copyDtoH, copyDtoH, end);
    cudaEventElapsedTime(&time_total, start, end);

    printf("Top-Down BFS Timings (ms): Copy H->D: %f, Kernel: %f, Copy D->H: %f, Total: %f\n",
           time_copyHtoD, time_kernel, time_copyDtoH, time_total);

    cudaFree(d_row_offsets);
    cudaFree(d_col_indices);
    cudaFree(d_distances);
    cudaFree(d_frontier_current);
    cudaFree(d_frontier_next);
    cudaFree(d_frontier_size);
    free(h_distances);

    cudaEventDestroy(start);
    cudaEventDestroy(copyHtoD);
    cudaEventDestroy(kernelStart);
    cudaEventDestroy(kernelEnd);
    cudaEventDestroy(copyDtoH);
    cudaEventDestroy(end);
}

void run_bfs_bottom_up(const int *h_row_offsets, const int *h_col_indices, int V, int source) {

    cudaEvent_t start, copyHtoD, kernelStart, kernelEnd, copyDtoH, end;
    cudaEventCreate(&start);
    cudaEventCreate(&copyHtoD);
    cudaEventCreate(&kernelStart);
    cudaEventCreate(&kernelEnd);
    cudaEventCreate(&copyDtoH);
    cudaEventCreate(&end);

    cudaEventRecord(start, 0);

    // Copy CSR graph to device
    int *d_row_offsets, *d_col_indices;
    int E = h_row_offsets[V];
    gpuErrchk( cudaMalloc((void**)&d_row_offsets, sizeof(int)*(V+1)) );
    gpuErrchk( cudaMalloc((void**)&d_col_indices, sizeof(int)*E) );
    gpuErrchk( cudaMemcpy(d_row_offsets, h_row_offsets, sizeof(int)*(V+1), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_col_indices, h_col_indices, sizeof(int)*E, cudaMemcpyHostToDevice) );

    // Allocate and initialize distances
    int *d_distances;
    gpuErrchk( cudaMalloc((void**)&d_distances, sizeof(int)*V) );
    gpuErrchk( cudaMemset(d_distances, 0x3f, sizeof(int)*V) );
    int zero = 0;
    gpuErrchk( cudaMemcpy(&d_distances[source], &zero, sizeof(int), cudaMemcpyHostToDevice) );

    cudaEventRecord(copyHtoD, 0);

    int level = 0;
    int h_changed;
    int *d_changed;
    gpuErrchk( cudaMalloc((void**)&d_changed, sizeof(int)) );

    cudaEventRecord(kernelStart, 0);
    do {
        h_changed = 0;
        gpuErrchk( cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice) );
        int num_blocks = (V + BLOCK_SIZE - 1) / BLOCK_SIZE;
        bfs_bottom_up_kernel<<<num_blocks, BLOCK_SIZE>>>(V, d_row_offsets, d_col_indices, level, d_distances, d_changed);
        gpuErrchk( cudaDeviceSynchronize() );
        gpuErrchk( cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost) );
        level++;
    } while(h_changed);
    cudaEventRecord(kernelEnd, 0);

    // Copy distances back to host
    int *h_distances = (int*)malloc(sizeof(int)*V);
    gpuErrchk( cudaMemcpy(h_distances, d_distances, sizeof(int)*V, cudaMemcpyDeviceToHost) );
    cudaEventRecord(copyDtoH, 0);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

    float time_copyHtoD, time_kernel, time_copyDtoH, time_total;
    cudaEventElapsedTime(&time_copyHtoD, start, copyHtoD);
    cudaEventElapsedTime(&time_kernel, kernelStart, kernelEnd);
    cudaEventElapsedTime(&time_copyDtoH, copyDtoH, end);
    cudaEventElapsedTime(&time_total, start, end);

    printf("Bottom-Up BFS Timings (ms): Copy H->D: %f, Kernel: %f, Copy D->H: %f, Total: %f\n",
           time_copyHtoD, time_kernel, time_copyDtoH, time_total);

    cudaFree(d_row_offsets);
    cudaFree(d_col_indices);
    cudaFree(d_distances);
    cudaFree(d_changed);
    free(h_distances);

    cudaEventDestroy(start);
    cudaEventDestroy(copyHtoD);
    cudaEventDestroy(kernelStart);
    cudaEventDestroy(kernelEnd);
    cudaEventDestroy(copyDtoH);
    cudaEventDestroy(end);
}

void run_bfs_direction_optimized(const int *h_row_offsets, const int *h_col_indices, int V, int source) {

    cudaEvent_t start, copyHtoD, kernelStart, kernelMiddle, kernelEnd, copyDtoH, end;
    cudaEventCreate(&start);
    cudaEventCreate(&copyHtoD);
    cudaEventCreate(&kernelStart);
    cudaEventCreate(&kernelMiddle);
    cudaEventCreate(&kernelEnd);
    cudaEventCreate(&copyDtoH);
    cudaEventCreate(&end);

    cudaEventRecord(start, 0);

    // Copy graph data
    int *d_row_offsets, *d_col_indices;
    int E = h_row_offsets[V];
    gpuErrchk( cudaMalloc((void**)&d_row_offsets, sizeof(int)*(V+1)) );
    gpuErrchk( cudaMalloc((void**)&d_col_indices, sizeof(int)*E) );
    gpuErrchk( cudaMemcpy(d_row_offsets, h_row_offsets, sizeof(int)*(V+1), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_col_indices, h_col_indices, sizeof(int)*E, cudaMemcpyHostToDevice) );

    // Allocate and initialize distances
    int *d_distances;
    gpuErrchk( cudaMalloc((void**)&d_distances, sizeof(int)*V) );
    gpuErrchk( cudaMemset(d_distances, 0x3f, sizeof(int)*V) );
    int zero = 0;
    gpuErrchk( cudaMemcpy(&d_distances[source], &zero, sizeof(int), cudaMemcpyHostToDevice) );

    // Allocate frontier arrays
    int *d_frontier_current, *d_frontier_next;
    gpuErrchk( cudaMalloc((void**)&d_frontier_current, sizeof(int)*V) );
    gpuErrchk( cudaMalloc((void**)&d_frontier_next, sizeof(int)*V) );
    gpuErrchk( cudaMemcpy(d_frontier_current, &source, sizeof(int), cudaMemcpyHostToDevice) );
    int frontier_size = 1;

    int *d_frontier_size;
    gpuErrchk( cudaMalloc((void**)&d_frontier_size, sizeof(int)) );

    cudaEventRecord(copyHtoD, 0);

    int level = 0;
    const int threshold = V / 10;
    cudaEventRecord(kernelStart, 0);
    // Top-down phase
    while(frontier_size > 0 && frontier_size < threshold) {
        gpuErrchk( cudaMemset(d_frontier_size, 0, sizeof(int)) );
        int num_blocks = (frontier_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        bfs_top_down_kernel<<<num_blocks, BLOCK_SIZE>>>(V, d_row_offsets, d_col_indices,
                                                          level, d_distances, d_frontier_current, frontier_size,
                                                          d_frontier_next, d_frontier_size);
        gpuErrchk( cudaDeviceSynchronize() );
        int new_frontier_size;
        gpuErrchk( cudaMemcpy(&new_frontier_size, d_frontier_size, sizeof(int), cudaMemcpyDeviceToHost) );
        frontier_size = new_frontier_size;
        int *temp = d_frontier_current;
        d_frontier_current = d_frontier_next;
        d_frontier_next = temp;
        level++;
    }
    cudaEventRecord(kernelMiddle, 0);
    // Bottom-up phase
    int h_changed;
    int *d_changed;
    gpuErrchk( cudaMalloc((void**)&d_changed, sizeof(int)) );
    while(true) {
        h_changed = 0;
        gpuErrchk( cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice) );
        int num_blocks = (V + BLOCK_SIZE - 1) / BLOCK_SIZE;
        bfs_bottom_up_kernel<<<num_blocks, BLOCK_SIZE>>>(V, d_row_offsets, d_col_indices, level, d_distances, d_changed);
        gpuErrchk( cudaDeviceSynchronize() );
        gpuErrchk( cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost) );
        if(h_changed == 0) break;
        level++;
    }
    cudaEventRecord(kernelEnd, 0);

    // Copy distances back to host
    int *h_distances = (int*)malloc(sizeof(int)*V);
    gpuErrchk( cudaMemcpy(h_distances, d_distances, sizeof(int)*V, cudaMemcpyDeviceToHost) );
    cudaEventRecord(copyDtoH, 0);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

    float time_copyHtoD, time_kernel, time_copyDtoH, time_total;
    cudaEventElapsedTime(&time_copyHtoD, start, copyHtoD);
    cudaEventElapsedTime(&time_kernel, kernelStart, kernelEnd);
    cudaEventElapsedTime(&time_copyDtoH, copyDtoH, end);
    cudaEventElapsedTime(&time_total, start, end);

    printf("Direction-Optimized BFS Timings (ms): Copy H->D: %f, Kernel: %f, Copy D->H: %f, Total: %f\n",
           time_copyHtoD, time_kernel, time_copyDtoH, time_total);

    cudaFree(d_row_offsets);
    cudaFree(d_col_indices);
    cudaFree(d_distances);
    cudaFree(d_frontier_current);
    cudaFree(d_frontier_next);
    cudaFree(d_frontier_size);
    cudaFree(d_changed);
    free(h_distances);

    cudaEventDestroy(start);
    cudaEventDestroy(copyHtoD);
    cudaEventDestroy(kernelStart);
    cudaEventDestroy(kernelMiddle);
    cudaEventDestroy(kernelEnd);
    cudaEventDestroy(copyDtoH);
    cudaEventDestroy(end);
}

void run_bfs_edge_centric(const int *h_src, const int *h_dst, int V, int E, int source) {

    cudaEvent_t start, copyHtoD, kernelStart, kernelEnd, copyDtoH, end;
    cudaEventCreate(&start);
    cudaEventCreate(&copyHtoD);
    cudaEventCreate(&kernelStart);
    cudaEventCreate(&kernelEnd);
    cudaEventCreate(&copyDtoH);
    cudaEventCreate(&end);

    cudaEventRecord(start, 0);

    // Copy COO graph to device
    int *d_edge_src, *d_edge_dst;
    gpuErrchk( cudaMalloc((void**)&d_edge_src, sizeof(int)*E) );
    gpuErrchk( cudaMalloc((void**)&d_edge_dst, sizeof(int)*E) );
    gpuErrchk( cudaMemcpy(d_edge_src, h_src, sizeof(int)*E, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_edge_dst, h_dst, sizeof(int)*E, cudaMemcpyHostToDevice) );

    // Allocate and initialize distances
    int *d_distances;
    gpuErrchk( cudaMalloc((void**)&d_distances, sizeof(int)*V) );
    gpuErrchk( cudaMemset(d_distances, 0x3f, sizeof(int)*V) );
    int zero = 0;
    gpuErrchk( cudaMemcpy(&d_distances[source], &zero, sizeof(int), cudaMemcpyHostToDevice) );

    cudaEventRecord(copyHtoD, 0);

    int level = 0;
    int h_changed;
    int *d_changed;
    gpuErrchk( cudaMalloc((void**)&d_changed, sizeof(int)) );

    cudaEventRecord(kernelStart, 0);
    while(true) {
        h_changed = 0;
        gpuErrchk( cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice) );
        int num_blocks = (E + BLOCK_SIZE - 1) / BLOCK_SIZE;
        bfs_edge_centric_kernel<<<num_blocks, BLOCK_SIZE>>>(E, d_edge_src, d_edge_dst, level, d_distances, d_changed);
        gpuErrchk( cudaDeviceSynchronize() );
        gpuErrchk( cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost) );
        if(h_changed == 0) break;
        level++;
    }
    cudaEventRecord(kernelEnd, 0);

    // Copy distances back
    int *h_distances = (int*)malloc(sizeof(int)*V);
    gpuErrchk( cudaMemcpy(h_distances, d_distances, sizeof(int)*V, cudaMemcpyDeviceToHost) );
    cudaEventRecord(copyDtoH, 0);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

    float time_copyHtoD, time_kernel, time_copyDtoH, time_total;
    cudaEventElapsedTime(&time_copyHtoD, start, copyHtoD);
    cudaEventElapsedTime(&time_kernel, kernelStart, kernelEnd);
    cudaEventElapsedTime(&time_copyDtoH, copyDtoH, end);
    cudaEventElapsedTime(&time_total, start, end);

    printf("Edge-Centric BFS Timings (ms): Copy H->D: %f, Kernel: %f, Copy D->H: %f, Total: %f\n",
           time_copyHtoD, time_kernel, time_copyDtoH, time_total);

    cudaFree(d_edge_src);
    cudaFree(d_edge_dst);
    cudaFree(d_distances);
    cudaFree(d_changed);
    free(h_distances);

    cudaEventDestroy(start);
    cudaEventDestroy(copyHtoD);
    cudaEventDestroy(kernelStart);
    cudaEventDestroy(kernelEnd);
    cudaEventDestroy(copyDtoH);
    cudaEventDestroy(end);
}

int main() {
    // Sample graph (undirected, unweighted)
    // 7 vertices and 12 edges in CSR format.
    // Vertex 0: neighbors 1,2  
    // Vertex 1: neighbors 0,3,4  
    // Vertex 2: neighbors 0,5  
    // Vertex 3: neighbor 1  
    // Vertex 4: neighbors 1,6  
    // Vertex 5: neighbor 2  
    // Vertex 6: neighbor 4
    int V = 7;
    int h_row_offsets[] = {0, 2, 5, 7, 8, 10, 11, 12};
    int h_col_indices[] = {1, 2, 0, 3, 4, 0, 5, 1, 1, 6, 2, 4};

    // COO representation for edge-centric BFS.
    int E = 12;
    int h_src[] = {0, 0, 1, 1, 1, 2, 2, 3, 4, 4, 5, 6};
    int h_dst[] = {1, 2, 0, 3, 4, 0, 5, 1, 1, 6, 2, 4};

    int source = 0;

    run_bfs_top_down(h_row_offsets, h_col_indices, V, h_row_offsets[V], source);
    run_bfs_bottom_up(h_row_offsets, h_col_indices, V, source);
    run_bfs_direction_optimized(h_row_offsets, h_col_indices, V, source);
    run_bfs_edge_centric(h_src, h_dst, V, E, source);

    return 0;
}
