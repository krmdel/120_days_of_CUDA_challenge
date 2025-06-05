Day 94: Implementation of double threshold with edge tracking in CUDA

1) Summary of the daily tutorial

The code implements double threshold hysteresis in CUDA. Starting from a pre-computed magnitude image, every pixel is first classified as strong, weak or non-edge by comparing its magnitude to two thresholds. Next, the algorithm iteratively “grows” strong edges: every weak pixel that touches at least one strong neighbour is promoted to strong. When no more pixels change state, all remaining weak pixels are discarded. The result is a clean, one-pixel-wide binary edge map.

The double threshold step is

```math
\mathrm{label}[i] =
\begin{cases}
2 & \mathrm{if\ } \mathrm{mag}[i] \ge T_{\mathrm{high}} \\[6pt]
1 & \mathrm{if\ } T_{\mathrm{low}} \le \mathrm{mag}[i] < T_{\mathrm{high}} \\[6pt]
0 & \mathrm{otherwise}
\end{cases}
```

and the hysteresis step repeatedly applies

```math
\text{if }\mathrm{label}[p]=1
\text{ and }\exists\,q\in\mathcal{N}(p):\mathrm{label}[q]=2
\;\;\Longrightarrow\;\;\mathrm{label}[p]\leftarrow 2
```

- CUDA kernels:
    - classifyKernel: Performs the double-threshold test for every pixel and stores labels 0 / 1 / 2. Massive parallel map with one thread per pixel.
    - hystIterKernel: Executes one hysteresis sweep: weak pixels that touch a strong neighbour become strong; others stay unchanged. A device-side flag counts promotions so the host can decide when to stop iterating. 16×16 thread blocks traverse the image; promotion recorded with atomicAdd.
    - convertStrongToBinary: Final pass that converts label 2 (strong) to binary 1 and everything else to 0. Simple element-wise transform.                                               |

2) Implementation

Compiling the code:

<pre>nvcc .\double_threshold.cu -o double_threshold</pre>

Running the code after compiling:

<pre>double_threshold</pre>

<pre>Generated random magnitude 4096×4096
CPU total                 : 248.329 ms
GPU H2D copy              : 14.5804 ms
GPU classify kernel       : 0.153856 ms
GPU hysteresis kernels    : 1.1961 ms   (iterations = 5)
GPU D2H copy              : 3.43408 ms
GPU total                 : 27.9955 ms
Pixel disagreement        : 0.0392199 %</pre>