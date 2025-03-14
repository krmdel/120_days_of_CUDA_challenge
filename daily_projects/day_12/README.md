Day 12: Histogram

1) Resources:

The lecture 13 of GPU Computing by Dr. Izzat El Hajj  
Lecture 13: https://youtu.be/BiYieuVwUbg?si=YEwPLnZiwb0gQG0G

2) Topics Covered:

- Parallel patterns: histogram
- New feature: atomic operations
- Optimization: privatization

3) Summary of the Lecture:  

- A histogram approximates the distribution of a dataset by:
    - Dividing the range of the dataset into bins
    - Counting the number of elements in each bin
- A data race occurs when multiple threads access the same memory location concurently without ordering and at least one access is a write. Data races may results in unpredictable program output. Example:

Thread A:

oldVal = bins[b]
newVal = oldVal + 1
bins[b] = newVal

Thread B:

oldVal = bins[b]
newVal = oldVal + 1
bins[b] = newVal

If both threads have the same b and bins[b] is initially 0, the final value of bins[b] could be 2 or 1.

- To avoid data races, concurrent read-modify-write operations to the same memory location need to be made mutually exclusive to enforce ordering.
- One way to do this on CPUs is using locks (mutex):

    mutex_lock(lock);
    ++bind[b];
    mutex_unlock(lock);

- Using locks with SIMD execution may cause deadlock: Assume threads 0 and 1 are in the same warp and try to acquire the same lock. Thread 0 acquires lock but Thread 1 waits for Thread 0 to release the lock. Moreover, Thread 0 waits for Thread 1 to complete previous SIMD.

- Atomic Operations:
    - perform read-modify-write with a single ISA instruction.
    - the hardware guarantees that no other thread can access the memory location until the operation completes.
    - Concurrent atomic opreations to the same memory location are serialized by the hardware.

- AtomicAdd:
    - T atomicAdd(T* address, T val) where T can be int, float, double. Reads the value stored at address, adds val to it, stores the new value to address and returns the old value originally stored
    - Function call translated to a single ISA instruction, such special functions are called intrinsics
    - Other atomic operations: sub, min, max, inc, dec, and, or, xor, exchange, compare and swap

- Atomic operations on global memory have high latency: need to wait for both read and write to complete, need to wait if there are other threads accessing the same location (high probability of contention) --> to optimize, use privatization!

- Privatization is an optimization where multiple private copies of an output are maintained, then the global copy is updated on completion
- Operations on the output must be associative and commutative because the order of the updates has changed
- Advantage:
    - Reduces contention on the global copy
    - If the output is small enough, the private copy can be placed in shared memory reducing access latency

- Example: histogram calculation
    - Each block updates a private copy of the histogram
    - Atomically update the global copy when done (once per nonzero bin per block)

- Thread coarsening:
    - Using fewer thread blocks result in fewer private copies, hence fewer global memory atomics to update the global copy
    - Apply thread coarsening to reduce the number of thread blocks and have each thread count multiple inputs: make sure to load input in coallesced way
    
4) Implementation:

The code performs reduction operation with coalescing and also for minimizing divergence as well as shared memory for further optimization on the GPU

Compiling the code:  

<pre>nvcc .\reduction_optim.cu -o reduceoptim</pre>

Running the code after compiling: 
<pre> reduceoptim </pre>

<pre>CPU Sum: 16777312.0000
GPU Sum: 16925198.0000

Timing Comparison:
CPU Time: 38.515 ms
GPU copy to GPU Time: 8.892 ms
GPU Kernel Time: 1.277 ms
GPU copy from GPU Time: 0.030 ms
GPU Total Time: 10.199 ms</pre>
