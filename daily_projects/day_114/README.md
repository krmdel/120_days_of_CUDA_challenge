Day 114: Implementation of feature detectors & descriptors in CUDA

1) Summary of the daily tutorial

The code implements a single scale Scale Invariant Feature Transform (SIFT) style pipeline. Starting from an 8-bit grayscale image, the code:
- converts pixels to 32-bit floats,
- builds three Gaussian-blurred images,
- computes two Differences-of-Gaussians (DoG) to isolate blobs,
- detects scale-space extrema as provisional key-points,
- assigns each key-point a dominant orientation from Sobel gradients, and
- packs a 128-D descriptor histogram for every surviving key-point.

   𝐷(𝑥,𝑦,𝜎)  =  𝐿(𝑥,𝑦,𝑘𝜎)  −  𝐿(𝑥,𝑦,𝜎)

(Difference-of-Gaussians at pixel (𝑥,𝑦); 𝐿 is the Gaussian-blurred image, 𝑘 is the blur ratio.)

- CUDA kernels:
   - u8ToF32: Copy & cast raw 8-bit image to float	global-memory read
   - gaussH: Separable 7-tap Gaussian blur (horizontal & vertical)
   - dogKer: Element-wise DoG
   - sobelKer: Per-pixel gradient magnitude & orientation
   - extremaKer: Detect extrema across three DoG layers & threshold
   - oriKer: Build 36-bin orientation histogram around each key-point, pick peak
   - descKer: Form 4×4 × 8 = 128-bin descriptor, L2-normalise

2) Implementation

Compiling the code:

<pre>nvcc .\feat.cu -o feat</pre>

Running the code after compiling:

<pre>feat</pre>

<pre>[Info] random 4K 3840×2160
Image 3840×2160
GPU keypoints          : 0
CPU keypoints          : 429186
GPU H2D copy           : 1.69882 ms
GPU kernels            : 0.00409603 ms
GPU total              : 1.70291 ms
CPU total              : 363.134 ms
Speed-up (CPU / GPU)   : 213.243×</pre>