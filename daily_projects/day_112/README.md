Day 112: Implementation of single frame HOG pipeline with dual streams in CUDA

1) Summary of the daily tutorial

The code computes Histogram-of-Oriented-Gradients (HOG) descriptor for a single grayscale frame, then benchmarks CPU vs. GPU performance.
   - Gradient stage (Sobel filter): magnitude M(x,y) and orientation theta(x,y)
   - Cell stage (8 × 8 pixels): 9-bin histogram per cell
   - Block stage (2 × 2 cells): 36-D descriptor with L2-Hys normalisation
  
  ```math
  \text{Magnitude } M(x,y)=\sqrt{G_x^2+G_y^2},\qquad
  \theta(x,y)=\operatorname{atan2}(G_y,G_x)\pmod{180^{\circ}}
  ```

  ```math
  b = \Bigl\lfloor \dfrac{\theta(x,y)}{20^{\circ}} \Bigr\rfloor,\qquad
  h_{cell}[b] \mathrel{+}= M(x,y)
  ```

  ```math
  \mathbf{v} \leftarrow \frac{\min\!\bigl(\mathbf{v}/\lVert\mathbf{v}\rVert_2,\; \text{clip}\bigr)}
                               {\sqrt{\lVert\min(\mathbf{v}/\lVert\mathbf{v}\rVert_2,\text{clip})\rVert_2^2+\varepsilon}}
  ```

CUDA kernels:
   - sobelKer: Loads a 16 × 16 tile plus halo into shared memory, computes G_x, G_y, then writes gradient magnitude and half-plane orientation.
   - cellKer: Accumulates per-pixel magnitudes into a 9-bin histogram for every 8 × 8 cell. Uses warp-level shuffle to sum bins across the 32 threads that cooperate on one cell.
   - blockKer: Gathers four neighbouring cell histograms, performs L2-Hys normalisation and clipping (0.2), outputs a 36-component block descriptor.
   - wSum: Single-warp reduction used by cellKer for fast histogram bin summation.

2) Implementation

Compiling the code:

<pre>nvcc .\hog_pipeline.cu -o hog_pipeline</pre>

Running the code after compiling:

<pre>hog_pipeline</pre>

<pre>CPU total               : 852.264 ms
GPU H2D copy            : 0.316992 ms
GPU kernels             : 0.946176 ms
GPU D2H copy            : 2.86118 ms
GPU total               : 4.12435 ms
RMSE descriptor         : 9.14997e-05</pre>