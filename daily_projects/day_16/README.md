Day 16: Sparse Matrix Computation (ELL and JDS)

1) Resources:

The lecture 17 of GPU Computing by Dr. Izzat El Hajj  
Lecture 17: https://youtu.be/bDbUoRrT6Js?si=pVWCuzC56OyQj1n-

2) Topics Covered:

- ELLPACK format (ELL)
- Jagged Diagonal Storage (JDS)

3) Summary of the Lecture:  

- ELLPACK format (ELL): 
- group nonzeros by row (like CSR)

Matrix:  
1 7 0 0  
5 0 3 9  
0 2 8 0  
0 0 0 6  

Column:     
0 1 *       
0 2 3       
1 2 *       
3 * *       

Value:
1 7 *  
5 3 9  
2 8 *  
6 * *  

- store padded array of nonzeros in column major order:  

Column: 0 0 1 3 1 2 2 * * 3 * *  
Value: 1 5 2 6 7 3 8 * * 9 * *

- Parallelization approach: Assign one thread to loop over each input row sequentially and update corresponding output element --> Memory acceseses are coalesced

Advantages of ELL:
- Flexibility: can add new elements as long as row not full
- Accessibility: given a row, easy to find all nonzeros; given nonzero, easy to find row and column
- SpMV/ELL memory accesses are coalesced

Disadvantages of ELL:
- Space efficiency: overhead tue to padding
- Accessibility: given a column, hard to find all nonzeros
- SpMV/ELL has control divergence

- Hybrid ELL/COO: ELL for dense rows, COO for very long rows
    - Similar to ELL, with the following added benefits from using COO:
        - Space efficiency: less padding
        - Flexibility: can add new elements to any row

- Jagged Diagonal Storage (JDS):
    - group nonzeros by row (like CSR)
    - sort the rows by size and remember the original row index
    - store nonzeros in column major order
Parallelization approach: Assign one thread to loop over each input row sequentially and update corresponding output element --> Memory acceseses are coalesced. Threads are drop out from the end, minimizing control divergence

Advantages of JDS:
- Space efficiency: no padding
- Accessibility: given a row, easy to find all nonzeros
- SpMV/JDS memory accesses are coalesced
- SpMV/JDS minimizes control divergence

Disadvantages of JDS:
- Flexibility: hard to add new elements to the matrix
- Accessibility: given nonzero, hard to find row; given a column, hard to find all nonzeros

4) Implementation:

The code performs merge on the GPU

Compiling the code:  

<pre>nvcc .\merge.cu -o merge</pre>

Running the code after compiling: 
<pre> merge </pre>

<pre>==== Baseline Kernel ====
CPU->GPU Copy Time: 0.06144 ms
Kernel Execution Time: 1.01171 ms
GPU->CPU Copy Time: 0.046176 ms
Total GPU Baseline Merge Time: 1.11933 ms
==== Optimized Kernel ====
CPU->GPU Copy Time: 0.033088 ms
Kernel Execution Time: 0.063488 ms
GPU->CPU Copy Time: 0.047424 ms
Total GPU Optimized Merge Time: 0.144 ms</pre>
