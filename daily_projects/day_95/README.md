Day 95: Implementation of structure tensor primitives in CUDA

1) Summary of the daily tutorial
        |

2) Implementation

Compiling the code:

<pre>nvcc .\tensor.cu -o tensor</pre>

Running the code after compiling:

<pre>tensor</pre>

<pre>Generated random gradients 4096×4096
CPU total                 : 1251.17 ms
GPU H2D copy              : 28.7524 ms
GPU kernels (all)         : 1.01414 ms
GPU D2H copy              : 42.7668 ms
GPU total                 : 173.662 ms
RMSE Ix² blurred (CPU/GPU): 1.07696e+06</pre>