Day 10: Scan (Kogge Stone)

1) Resources:

The lecture 11 of GPU Computing by Dr. Izzat El Hajj  
Lecture 11: https://youtu.be/-eoUw8fTy2E?si=bYEewVGxeaJrd2ES

2) Topics Covered:

- Parallel patterns: scan
    - Kogge-Stone method
- Optimization: double-buffering

3) Summary of the Lecture:  

- A scan operation takes an input array ([x0, x1, x2, ... , xn-1]) and associative operator (e.g., ⊕: sum, product, min, max) and returns an output array ([y0, y1. y2, ... , yn-1]). Two ways of scan:

Inclusive scan: yi = x0 ⊕ x1 ⊕ x2 ⊕ ... ⊕ xi
Exclusive scan: yi = x0 ⊕ x1 ⊕ x2 ⊕ ... ⊕ xi-1

- Sequential scan:

Inclusive scan:
output[0] = input[0];
for (int i = 1; i < n; i++) {
    output[i] = f(output[i-1], input[i]);
}

Exclusive scan:
output[0] = identity;
for (int i = 1; i < n; i++) {
    output[i] = f(output[i-1], input[i-1]);
}

- Segmented scan: 

    - Similar to reduction, threads must synchronize to perform scan, cannot synchronize across threads in different blocks.
    - Solution is segmented scan (or hierarchical scan): every thread blocks scan a segment, scan the partial sums, update the segments based on partial sums

- Optimizations: shared memory and double buffering (element is read by one thread and written by another thread, needs to make sure that thread writing does not interfere with the thread reading)

4) Implementation:

The code performs convolution operation over constant memory on the GPU

Compiling the code for vector addition:  

<pre>nvcc .\stencil.cu -o stencil</pre>

Running the code after compiling: 
<pre> stencil </pre>

<pre>CPU stencil 3D: 111.396477 ms
GPU stencil 3D: 1.225056 ms
GPU tiled stencil 3D: 0.157696 ms
GPU coarse stencil 3D: 0.265216 ms</pre>
