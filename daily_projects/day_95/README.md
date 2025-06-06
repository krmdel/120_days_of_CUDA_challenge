Day 95: Implementation of structure tensor primitives in CUDA

1) Summary of the daily tutorial

The code implements the core steps needed to form a 2-D structure tensor:

- Gradient products: Compute I_x^2, I_y^2 and I_x.I_y for every pixel.
- Separable 7 × 7 box blur: A pair of horizontal to vertical passes smooth each product with a radius-3 (7-tap) box filter, yielding the windowed sums that make up the tensor.

Gradient products:

```math
I_x^2 = I_x \cdot I_x,\qquad
I_y^2 = I_y \cdot I_y,\qquad
I_{xy} = I_x \cdot I_y
```

7-tap box blur (radius R=3) applied to any image f:

```math
\langle f \rangle(x,y) \mathrel{:=}
\sum_{i=-R}^{R} \sum_{j=-R}^{R} f(x+i,\,y+j)
```

Structure tensor at each pixel:

```math
T(x,y)=
\begin{bmatrix}
\langle I_x^2  \rangle & \langle I_x I_y \rangle\\
\langle I_x I_y \rangle & \langle I_y^2  \rangle
\end{bmatrix}
```

- CUDA kernels:
    - productKernel: Point-wise gradient products. One thread per pixel computes I_x^2, I_y^2, I_x.I_y in global memory.                                                           |
    - boxBlurH: Horizontal 7-tap blur. Each block covers one image row; threads load clamped neighbours [x - 3, x + 3], accumulate and write an intermediate result.
    - boxBlurV: Vertical 7-tap blur. Mirrors the horizontal pass down columns to finish the separable box
    filter.                                                         |

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