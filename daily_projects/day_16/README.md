Day 16: Sparse Matrix Computation (ELL and JDS)

1) Resources:

The lecture 17 of GPU Computing by Dr. Izzat El Hajj  
Lecture 17: https://youtu.be/bDbUoRrT6Js?si=pVWCuzC56OyQj1n-

2) Topics Covered:

- 

3) Summary of the Lecture:  

- 

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
