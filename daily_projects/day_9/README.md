Day 9: Reduction

1) Resources:

The lecture 10 of GPU Computing by Dr. Izzat El Hajj  
Lecture 10: https://youtu.be/voFt2e2QXtA?si=GGJLQzbAf0WcgWVL

2) Topics Covered:

- Parallel pattern: reduction

3) Summary of the Lecture:  

- A reduction operation reduces a set of input values to one value (e.g., sum, max, min)
- Reduction operation is associative, commutative and have a well-define identity value
- Sequential reduction
- Parallel reduction: Every thread adds two elements in each step. Takes log(n) steps and half the threads drop out every step. Pattern is called "reduction tree"
- Segmented reduction: adding partial sums together
- Coalescing and minimizing divergence
- Initial load from global memory. Subsequent writes and reads continue in shared memory
- Applying Thread Coarsening
- Coarsening benefits:
    - If blocks are all executed in parallel: log(n) steps, log(n) synchronization
    - If blocks serilized by the hardware by a factor of C: C*log(n) steps, C*log(n) synchronization
    - If blocks are coarsened by a factor of C: 2*(C-1) + log(n) steps, log(n) synchronization

4) Implementation:

The code performs convolution operation over constant memory on the GPU

Compiling the code for vector addition:  

<pre>nvcc convolution.cu -o conv</pre>

Running the code after compiling: 
<pre>conv</pre>

<pre>CPU Time: 19.318785 ms
GPU Time without constant memory: 2.294944 ms
GPU Time with constant memory: 0.531520 ms</pre>
