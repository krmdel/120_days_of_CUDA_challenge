Day 5: Performance Considerations

1) Resources:

The lecture 6 of GPU Computing by Dr. Izzat El Hajj  
Lecture 6: https://youtu.be/DA-_EK8PbTY?si=q0EFCXde5kNKxE5Z

2) Topics Covered:

- Architecture of DRAM
- More performance optimizations

3) Summary of the Lecture:  

- Performance optimizations covered in previous lectures:
    - Tuning resource usage to maximize occupancy: threads per block, shared memory per block, registers per thread
    - Minimizing control divergence to increase SIMD efficiency
    - Shared memory tiling to increase data reuse
- Further optimiations: 
    - Memory coalescing
    - Thread coarsening
- The structure of DRAM cell: it contains of a capacitance that stores a change and a three-state device (single transistor) that allows data to be read/written (capacitor discharge/charge)  
- Slow: DRAM array, Sense amps, Column Latches. Fast: Mux
- DRAM burst rapidly reads data at DRAM array to the column latches. 
    - Accesing the data in different bursts, need to access the array again.
    - Accessing the data in the same burst, no need to access the array again, just the multiplexer.
- When threds in the same warp access consecutive memory locations in the same burst, the accesses can be combined and served by one burst
    - One DRAM transaction is needed
    - Known as Memory Coalescing. Examples: vector addition, matrix multiplication
- If threads in the same warp access locations not in the same burst, accesses cannot be combined
    - Multiple DRAM transactions are needed, takes longer to service data to the warp
    - Known as Memory Divergence
- The lateny can be hidden by having multiple DRAM banks.  
    - Need mny threads to simultaneously access memory to keep all the banks busy (with high occupancy in SMs)
- Thread coarsening is an optimization where a thred is assigned multiple units ot parallelism to process
    - Advantage: Redundant memory lodas in the tiled matrix multiplication, redundant computations, synchronization overheads or divergence
    - Disadvantage: Underutilization of resources if coarsening factor is too high (needs to be fine-tuned for each device), more resources per thread which may limit occupancy

- Tension Between Optimizations:  
    - Maximizing occupancy hides pipeline latency, bit too many threads may compete for the cache, evicting each other's data (thrashing the cache).
    - Shared memory tiling enables more data reuse but may limit occupancy
    - Thread coarsening reduces redundant work, but requires more resources per thread which may limit occupancy

- To find the sweetspot, identify what the botleneck is!  

4) Implementation:

The code performs tiled matrix multiplication on the CPU/GPU (without thread sync)

Compiling the code:  

<pre> nvcc matmultiled.cu -o matmultiled </pre>

Running the code after compiling: 
<pre> matmultiled </pre>

The code was executed on the NVIDIA GeForce RTX 3080Ti Laptop GPU - 16 GB. The output of the code is as follows:

<pre>======== CPU Tiled (No Shared) ========
Alloc time (ms):    0.012
Copy/init time (ms):2.0838
Kernel time (ms):   2602.39
Total CPU time (ms):2604.48

======== GPU Tiled (NO Shared) ========
Alloc time (ms):       1.1817
Host->Device (ms):     0.883104
Kernel time (ms):      3.40848
Device->Host (ms):     0.502112
Total GPU time (ms):   5.97539

======== GPU Tiled (WITH Shared) ========
Alloc time (ms):       6.11632
Host->Device (ms):     0.880768
Kernel time (ms):      1.53286
Device->Host (ms):     0.450336
Total GPU time (ms):   8.98029
</pre>
