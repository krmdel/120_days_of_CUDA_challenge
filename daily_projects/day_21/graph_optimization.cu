#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define INF 0x3f3f3f3f
#define BLOCK_SIZE 256

#define BLOCK_QUEUE_SIZE 1024

#define SINGLE_BLOCK_QUEUE_SIZE 1024

// Macro to check for CUDA errors.
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

__global__ void bfs_optimized_kernel(int V, const int *row_offsets, const int *col_indices,
                                       int current_level, int *distances,
                                       const int *frontier, int frontier_size,
                                       int *next_frontier, int *next_frontier_size) {
    extern __shared__ int s_queue[];  // shared queue for this block, size = BLOCK_QUEUE_SIZE
    __shared__ int s_queue_size;
    if (threadIdx.x == 0) {
        s_queue_size = 0;
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < frontier_size; i += stride) {
         int u = frontier[i];
         for (int offset = row_offsets[u]; offset < row_offsets[u+1]; offset++) {
              int v = col_indices[offset];
              if (atomicCAS(&distances[v], INF, current_level + 1) == INF) {
                  int pos = atomicAdd(&s_queue_size, 1);
                  if (pos < BLOCK_QUEUE_SIZE) {
                      s_queue[pos] = v;
                  } else {
                      int global_pos = atomicAdd(next_frontier_size, 1);
                      next_frontier[global_pos] = v;
                  }
              }
         }
    }
    __syncthreads();

    if (threadIdx.x == 0 && s_queue_size > 0) {
         int base = atomicAdd(next_frontier_size, s_queue_size);
         for (int i = 0; i < s_queue_size; i++) {
              next_frontier[base + i] = s_queue[i];
         }
    }
}

__global__ void bfs_optimized_single_block_kernel(int V, const int *row_offsets, const int *col_indices,
                                                    int *distances, int source) {
    __shared__ int queue[SINGLE_BLOCK_QUEUE_SIZE];
    __shared__ int queue_size;
    int level = 0;

    if (threadIdx.x == 0) {
       queue[0] = source;
       queue_size = 1;
       distances[source] = 0;
    }
    __syncthreads();

    while (queue_size > 0) {
       __shared__ int next_queue[SINGLE_BLOCK_QUEUE_SIZE];
       __shared__ int next_queue_size;
       if (threadIdx.x == 0) {
           next_queue_size = 0;
       }
       __syncthreads();

       for (int i = threadIdx.x; i < queue_size; i += blockDim.x) {
           int u = queue[i];
           for (int offset = row_offsets[u]; offset < row_offsets[u+1]; offset++) {
                int v = col_indices[offset];
                if (distances[v] == INF) {
                    distances[v] = level + 1;
                    int pos = atomicAdd(&next_queue_size, 1);
                    if (pos < SINGLE_BLOCK_QUEUE_SIZE) {
                        next_queue[pos] = v;
                    }
                }
           }
       }
       __syncthreads();
       if (threadIdx.x == 0) {
           queue_size = next_queue_size;
           for (int i = 0; i < queue_size; i++) {
              queue[i] = next_queue[i];
           }
           level++;
       }
       __syncthreads();
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

    // BFS loop
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

    // Cleanup
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

    int *d_row_offsets, *d_col_indices;
    int E = h_row_offsets[V];
    gpuErrchk( cudaMalloc((void**)&d_row_offsets, sizeof(int)*(V+1)) );
    gpuErrchk( cudaMalloc((void**)&d_col_indices, sizeof(int)*E) );
    gpuErrchk( cudaMemcpy(d_row_offsets, h_row_offsets, sizeof(int)*(V+1), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_col_indices, h_col_indices, sizeof(int)*E, cudaMemcpyHostToDevice) );

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

    int *d_row_offsets, *d_col_indices;
    int E = h_row_offsets[V];
    gpuErrchk( cudaMalloc((void**)&d_row_offsets, sizeof(int)*(V+1)) );
    gpuErrchk( cudaMalloc((void**)&d_col_indices, sizeof(int)*E) );
    gpuErrchk( cudaMemcpy(d_row_offsets, h_row_offsets, sizeof(int)*(V+1), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_col_indices, h_col_indices, sizeof(int)*E, cudaMemcpyHostToDevice) );

    int *d_distances;
    gpuErrchk( cudaMalloc((void**)&d_distances, sizeof(int)*V) );
    gpuErrchk( cudaMemset(d_distances, 0x3f, sizeof(int)*V) );
    int zero = 0;
    gpuErrchk( cudaMemcpy(&d_distances[source], &zero, sizeof(int), cudaMemcpyHostToDevice) );

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

    int *d_edge_src, *d_edge_dst;
    gpuErrchk( cudaMalloc((void**)&d_edge_src, sizeof(int)*E) );
    gpuErrchk( cudaMalloc((void**)&d_edge_dst, sizeof(int)*E) );
    gpuErrchk( cudaMemcpy(d_edge_src, h_src, sizeof(int)*E, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_edge_dst, h_dst, sizeof(int)*E, cudaMemcpyHostToDevice) );

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

void run_bfs_optimized_multiblock(const int *h_row_offsets, const int *h_col_indices, int V, int E, int source) {

    cudaEvent_t start, copyHtoD, kernelStart, kernelEnd, copyDtoH, end;
    cudaEventCreate(&start);
    cudaEventCreate(&copyHtoD);
    cudaEventCreate(&kernelStart);
    cudaEventCreate(&kernelEnd);
    cudaEventCreate(&copyDtoH);
    cudaEventCreate(&end);

    cudaEventRecord(start, 0);

    // Allocate and copy CSR graph to device
    int *d_row_offsets, *d_col_indices;
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
    cudaEventRecord(kernelStart, 0);
    while(frontier_size > 0) {
        gpuErrchk( cudaMemset(d_frontier_size, 0, sizeof(int)) );

        int num_blocks = (frontier_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        size_t shared_mem_size = BLOCK_QUEUE_SIZE * sizeof(int);
        bfs_optimized_kernel<<<num_blocks, BLOCK_SIZE, shared_mem_size>>>(V, d_row_offsets, d_col_indices,
                                                       level, d_distances,
                                                       d_frontier_current, frontier_size,
                                                       d_frontier_next, d_frontier_size);
        gpuErrchk( cudaDeviceSynchronize() );

        gpuErrchk( cudaMemcpy(&frontier_size, d_frontier_size, sizeof(int), cudaMemcpyDeviceToHost) );

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

    printf("Optimized Multi-Block BFS Timings (ms): Copy H->D: %f, Kernel: %f, Copy D->H: %f, Total: %f\n",
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

void run_bfs_optimized_singleblock(const int *h_row_offsets, const int *h_col_indices, int V, int E, int source) {

    cudaEvent_t start, copyHtoD, kernelStart, kernelEnd, copyDtoH, end;
    cudaEventCreate(&start);
    cudaEventCreate(&copyHtoD);
    cudaEventCreate(&kernelStart);
    cudaEventCreate(&kernelEnd);
    cudaEventCreate(&copyDtoH);
    cudaEventCreate(&end);

    // Allocate and copy graph data
    int *d_row_offsets, *d_col_indices;
    gpuErrchk( cudaMalloc((void**)&d_row_offsets, sizeof(int)*(V+1)) );
    gpuErrchk( cudaMalloc((void**)&d_col_indices, sizeof(int)*E) );
    gpuErrchk( cudaMemcpy(d_row_offsets, h_row_offsets, sizeof(int)*(V+1), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_col_indices, h_col_indices, sizeof(int)*E, cudaMemcpyHostToDevice) );

    // Allocate distances array
    int *d_distances;
    gpuErrchk( cudaMalloc((void**)&d_distances, sizeof(int)*V) );

    cudaEventRecord(copyHtoD, 0);

    const int iterations = 1000;
    float total_kernel_time = 0.0f;
    for (int iter = 0; iter < iterations; iter++) {
        gpuErrchk( cudaMemset(d_distances, 0x3f, sizeof(int)*V) );
        int zero = 0;
        gpuErrchk( cudaMemcpy(&d_distances[source], &zero, sizeof(int), cudaMemcpyHostToDevice) );

        cudaEventRecord(kernelStart, 0);
        // Launch the single-block kernel
        bfs_optimized_single_block_kernel<<<1, BLOCK_SIZE>>>(V, d_row_offsets, d_col_indices, d_distances, source);
        gpuErrchk( cudaDeviceSynchronize() );
        cudaEventRecord(kernelEnd, 0);
        cudaEventSynchronize(kernelEnd);
        float iter_kernel_time;
        cudaEventElapsedTime(&iter_kernel_time, kernelStart, kernelEnd);
        total_kernel_time += iter_kernel_time;
    }
    float avg_kernel_time = total_kernel_time / iterations;

    // Copy distances back
    int *h_distances = (int*)malloc(sizeof(int)*V);
    gpuErrchk( cudaMemcpy(h_distances, d_distances, sizeof(int)*V, cudaMemcpyDeviceToHost) );
    cudaEventRecord(copyDtoH, 0);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float time_copyHtoD, time_copyDtoH, time_total;
    cudaEventElapsedTime(&time_copyHtoD, start, copyHtoD);
    cudaEventElapsedTime(&time_copyDtoH, copyDtoH, end);
    cudaEventElapsedTime(&time_total, start, end);

    printf("Optimized Single-Block BFS Timings (ms): Copy H->D: %f, Avg Kernel: %f, Copy D->H: %f, Total: %f\n",
           time_copyHtoD, avg_kernel_time, time_copyDtoH, time_total);

    cudaFree(d_row_offsets);
    cudaFree(d_col_indices);
    cudaFree(d_distances);
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

    run_bfs_optimized_multiblock(h_row_offsets, h_col_indices, V, h_row_offsets[V], source);
    run_bfs_optimized_singleblock(h_row_offsets, h_col_indices, V, h_row_offsets[V], source);

    return 0;
}
