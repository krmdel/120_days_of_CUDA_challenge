Day 24: Implementation for dynamic parallelism in CUDA

1) Implementation:

The code performs vector addition using pinned memory and streams on the GPU

Compiling the code:  

<pre>nvcc -rdc=true -arch=compute_50 -code=sm_50 dynamic_parallelism.cu -o dynamic</pre>

Running the code after compiling: 
<pre> dynamic </pre>

<pre>Time for CPU to GPU copy: 0.448864 ms
Time for kernel execution: 6.744032 ms
Time for GPU to CPU copy: 0.444064 ms
Total time: 7.680000 ms</pre>
