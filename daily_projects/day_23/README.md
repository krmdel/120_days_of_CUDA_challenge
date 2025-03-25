Day 24: Implementation of pinned memory and streams on the GPU

1) Implementation:

The code performs vector addition using pinned memory and streams on the GPU

Compiling the code:  

<pre>nvcc pinnedmemery_streams.cu -o pinstream</pre>

Running the code after compiling: 
<pre> pinstream </pre>

<pre>Streaming (pinned memory) timings:
Max host to device copy time per stream: 0.856608 ms
Max kernel execution time per stream: 0.104032 ms
Max device to host copy time per stream: 0.112480 ms
Max total time per stream: 1.008480 ms
Overall elapsed time: 1.466272 ms
Baseline timings:
Host to device copy time: 1.040832 ms
Kernel execution time: 0.059936 ms
Device to host copy time: 0.879392 ms
Total time: 1.981856 ms</pre>
