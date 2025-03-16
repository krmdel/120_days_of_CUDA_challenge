Day 14: Sort

1) Resources:

The lecture 15 of GPU Computing by Dr. Izzat El Hajj  
Lecture 15: https://youtu.be/XTfH6Ll9KaA?si=pquqTCy7rfsY3W1Z

2) Topics Covered:

- Parallel patterns: sort
    - Radix sort
    - Merge sort

3) Summary of the Lecture:  

Radix Sort:
- Radix sort is a sorting algorithm that distributes keys into buckets based on a radix (base)
- Distributing keys into buckets is repeated for each digit, while preserving the order from previous iterations within each bucket
- Using a radix that is a power of 2 simpliifies processing binary numbers

Example:
- Separate the keys into two buckets based on the first bit (the least significant bit)
- Next, separate the keys based on the second bit
- Preserving the order from previous iterations within each bucket ensures that keys are now sorted by the lower two bits
- Repeat the process for all bits
- Finding destination index:
    - destination of a zero = # zeros to the left = # elements to the left - # ones to the left = elemnt index - # ones to the left
    - destination of a one = # zeros in total + # ones to the left = (# elements in total - # ones in total) + # ones to the left = input size - # ones in total + # ones to the left
    - Need to find: # ones to the left of each element --> use exclusive scan
    - Parallelization approach: Assign one thread to each element --> Poor memory coalescing: stores are not coalesced (nearby threads write to distant locations in global memory)
    - Optimization: sort locally in shared memory, then write each bucket to global memory in a coalesced manner
- A larger radix can also be used: advantage: fewer iterations, disadvantage: more buckets --> results in poorer coalescing
- Choice of radix value must balance between the number of iterations and the coalescing behavior

- Thread coarsening:
    - The price of parallelizing across more blocks is having smaller buckets per block, hence fewer opportunities for coalescing
    - Processing more elements pre block results in larger buckets per block, hence better coalescing

Merge Sort:
- Radix sort is not applicable to all kinds of keys
- An alternative comparison-based sort that is highly amenable to parallelization is merge sort which divices the list into sublists, sorts the sublists, and then merges them (divide and conquer algorithm)
- Parallelization approach: Each step performs different merge operations in parallel, and also parallelizes each merge operation
    - Earlier steps rely more on parallelism across merge operations
    - Later steps rely more on parallelism within merge operations

4) Implementation:

The code performs scan using Brent-Kung method on the GPU

Compiling the code:  

<pre>nvcc .\scan_brentkung.cu -o scan</pre>

Running the code after compiling: 
<pre> scan </pre>

<pre>Brent-Kung Scan Kernel Time: 0.084 ms</pre>
