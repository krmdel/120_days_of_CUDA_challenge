Day 15: Sparse Matrix Computation (COO and CSR)

1) Resources:

The lecture 16 of GPU Computing by Dr. Izzat El Hajj  
Lecture 16: https://youtu.be/H6YGKNukGMo?si=o8ihOVdvaY19rrUK

2) Topics Covered:

- Parallel Patterns: Sparse Matrix Computation
- Case study: Sparse Matrix-Vector Multiplication (SpMV)
- Coordinate Format (COO)
- Compressed Sparse Row (CSR)

3) Summary of the Lecture:  

- A dense matrix is one where the majority of elements are non-zero. 
- A sparse matrix is one where the majority of elements are zero (many real world matrices are sparse)
- Opportunuities:
    - Do not need to allocate space for zeros (save memory)
    - Do not load zeros (save memory bandwidth)
    - Do not need to compute with zeros (save computation time)
- Sparse Matrix Storage Formats:
    - Coordinate Format (COO)
    - Compressed Sparse Row (CSR)
    - ELLPACK (ELL)
    - Jagged Diagonal Storage (JDS)
- Format design considerations:
    - Storage efficiency (memory consumed)
    - Flexibility (ease of adding/reordering elements)
    - Accessibility (ease of finding desired data)
    - Memory access pattern (enabling coalescing)
    - Load balance (minimizing control divergence)

+ Coordinate Format (COO):
    - Store every nonzero along with its row and column index
    
Matrix:

1 7 0 0
5 0 3 9
0 2 8 0
0 0 0 6

Row: 0 0 1 1 1 2 2 3
Col: 0 1 0 2 3 1 2 3
Val: 1 7 5 3 9 2 8 6

- Parallelization approach: Assign one thread to nonzero element
- Multiple threads can write to the same location in the output vector --> need atomic operations

Advantages of COO:
- Flexibility: easy to add new elements to the matrix, nonzeros can be stored in any order
- Accessibility: given nonzero, easy to find its row and column index
- SpMV/COO has coalesced memory access pattern
- SpMV/COO has no control divergence

Disadvantages of COO:
- Accessibility: given a row or column, hard to find all nonzeros (need to search)
- SpMV/COO uses atomic operations

+ Compresed Sparse Row (CSR):
    - Store nonzeros of the same row adjacently and an index to the first element of each row

Matrix:

1 7 0 0
5 0 3 9
0 2 8 0
0 0 0 6

RowPtrs: 0 2 5 7 (8)
ColInd: 0 1 0 2 3 1 2 3
Val: 1 7 5 3 9 2 8 6

Paralellization approach: Assign one thread to loop over each input row sequentially and update corresponding output element

Advantages of CSR:
- Space efficiency: row pointers smaller than row indexes
- Accessibility: given a row, easy to find all nonzeros
- SpMV/CSR avoids atomics, every thread owns its output

Disadvantages of CSR:
- Flexibility: hard to add new elements to the matrix
- Accessibility: given a nonzero, hard to find row; given a column, hard to find all nonzeros
- SpMV/CSR memory accesses are not coalesced
- SpMV/CSR has control divergence



4) Implementation:

The code performs histogram with privatization and shared memory and also thread coarsening on the GPU

Compiling the code:  

<pre>nvcc .\histogram.cu -o hist</pre>

Running the code after compiling: 
<pre> hist </pre>

<pre>=== Privatization Kernel Timing (ms) ===
Copy H2D: 0.187936 ms
Kernel Execution: 0.953056 ms
Copy D2H: 0.049472 ms
Total GPU Time: 1.19046 ms
CPU Histogram Time: 1.7001 ms

=== Thread Coarsening Kernel Timing (ms) ===
Copy H2D: 0.184096 ms
Kernel Execution: 0.102304 ms
Copy D2H: 0.03792 ms
Total GPU Time: 0.32432 ms</pre>
