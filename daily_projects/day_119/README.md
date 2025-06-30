Day 119: Block-matching stereo with 7×7 census transform 

1) Summary of the daily tutorial


The code implements a dense disparity map from a rectified stereo pair using census-transform block matching on the GPU.
- Each image is first converted to a 64-bit census signature per pixel over a 7 × 7 window.
- For every pixel in the left image, the code searches along the same scan-line in the right image up to 64 pixels to find the disparity with the smallest Hamming distance between census signatures.
- Performance for every stage—host↔device copies, census transform, matching, and the full pipeline—is timed and compared to a pure-CPU reference.

```math
\text{Census}(x,y) = \sum_{(dx,dy)\in\Omega\setminus(0,0)}
    \Bigl[\,I(x+dx,y+dy) < I(x,y)\Bigr]\;2^{b(dx,dy)}
```

```math
\text{cost}(d) = \operatorname{Hamming}\bigl(
      \text{Census}_\text{L}(x,y),\;
      \text{Census}_\text{R}(x-d,y)
\bigr)
```

- CUDA kernels:
  - censusKer: Calculates the 64-bit census descriptor for each valid pixel. Loads the centre pixel, iterates over the 48 neighbours, packs comparisons into a u64. Out-of-border pixels get a zero signature to avoid conditional branches later.
  - matchKer: For every pixel in the left image, iteratively compares its census signature with shifted signatures from the right image. Uses __popcll() to compute Hamming distance (population count) quickly. Keeps track of the best (lowest-cost) disparity up to MAX_DISP. Writes the winning disparity to the output disparity map.

2) Implementation

Compiling the code:

<pre>nvcc .\stereo_matching.cu -o stereo_matching</pre>

Running the code after compiling:

<pre>stereo_matching</pre>

<pre>[Info] random stereo pair 4096×4096

Image size              : 4096 × 4096
Search range            : 0-64 px

GPU H2D copy            : 7.4681 ms
GPU census kernels      : 1.03606 ms
GPU match kernel        : 1.80326 ms
GPU total               : 16.1792 ms
CPU total               : 6986.96 ms</pre>