Day 11: Scan (Brent Kung)

1) Resources:

The lecture 12 of GPU Computing by Dr. Izzat El Hajj  
Lecture 12: https://youtu.be/CcwdWP44aFE?si=7VZLbqBsKuxFHoDG

2) Topics Covered:

- Brent-Kung method of scan for parallel patterns
- Coarsening for scan

3) Summary of the Lecture:  

- Brent-Kung method is an inclusive parallel scan algorithm (one thread for every two elements)
- There are two stages: reduction step (adds every other element from its left) and post-reduction step (adds missing elements)
- Comparison between Kogge-Stone and Brent-Kung:
    - Brent-Kung has more steps but less operation per steps and in total compared to Kogge-Stone
    - Work efficiency:
         Kogge-Stone: log(n) steps, O(n*log(n)) operations
         Brent-Kung: 1) Reduction step: log(n) steps, n-1 operations, 2) Post-reduction step: log(n)-1 steps, (n-2) - (log((n)-1)) operations. Total: 2*log(n) - 1 steps, O(n) operations

- Brent-Kung takes more step but is more work-efficient
- Optimizations: 
    - Using shared memory similar to Kogge-Stone but also enables coalescing of glocal memory loads where data is already coalesced  
    - No need for double buffering unlike Kogge-Stone because no data element is read and written by different threads on the same iteration  
    - Control divergence is not a problem in Kogge-Stone as all the active threads are next to each other. However, in Brent-Kung, in every iteration, not all adjacent threads are active. Therefore, instead of assigning threads to specific data elements, re-indexing threads on every iteration to different elements (if there are M operations, assign them to the first M threads based on the thread index and stride value)

- Parallelizing scan incurs the overhead of lowering work efficiency. If reseources are insufficient, the hardware will serialize the thread blocks, incurring overhead unnecessarily. 

- Applying thread coarseining via segmented scan: 
    - Each thread scans a segment sequentially: Sequential scan is work efficient
    - Only scan the partial sums of each segment in parallel

4) Implementation:

The code performs reduction operation on the GPU

Compiling the code for vector addition:  

<pre>nvcc .\reduction.cu -o reduction</pre>

Running the code after compiling: 
<pre> reduction </pre>

<pre>CPU Sum: 16777312.0000
GPU Sum: 16793994.0000

Timing Comparison:
CPU time: 34.964 ms
GPU copy to device: 8.313 ms
GPU kernel time: 1.402 ms
GPU copy from device: 1.476 ms
GPU total time: 11.450 ms</pre>
