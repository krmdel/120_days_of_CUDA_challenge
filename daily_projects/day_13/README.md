Day 13: Merge

1) Resources:

The lecture 14 of GPU Computing by Dr. Izzat El Hajj  
Lecture 14: https://youtu.be/szoc52lNufU?si=Wp9wbVC8mbgYG2nk

2) Topics Covered:

- Parallel pattern: Merge

3) Summary of the Lecture:  

- An ordered merge operations takes two oredered lists and combines them into a single oredered list
- Parallelization approach: Divide the output into equal size segmenets and assign a thread to perform a sequential merge of each segment

A: m elements
B: n elements
C: m+n elements

Given k, finding i and j: i and j as the co-ranks of k:

k = i + j

max(0, k-n) <= i <= min(m, k) --> perfrom binary search within bound

A[i-1] <= B[j] and B[j-1] <= A[i]

- Memory accesses are not coalesced. During co-rank function, each thread performs binary search which has random access. During sequential merge, each thread loops through its own segment of consecutive elements. 
- Optimization: 
    - Load the entire block's segment to shared memory. Loads from global memory are coalesced. One thread in block does co-rank to find block's input segments.
    - Do the per-thread co-rank and merge in shared memory. Non-coalesced accesses performed in shared memory.

4) Implementation:

The code performs scan using Kogge-Stone method on the GPU

Compiling the code:  

<pre>nvcc .\scan_koggestone.cu -o scan</pre>

Running the code after compiling: 
<pre> scan </pre>

<pre>Baseline GPU Kernel Time: 0.739 ms
Shared Single Buffer GPU Kernel Time: 0.050 ms
Shared Double Buffer GPU Kernel Time: 0.034 ms</pre>
