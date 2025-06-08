Day 97: Implementation of FAST corner detector with warp-shuffle in CUDA

1) Summary of the daily tutorial

The code implements the FAST-9 corner detector, then compares speed and bit-exactness. A single warp-wide kernel that accelerates the same logic with:

- constant-memory lookup tables for circle offsets and every 9-pixel contiguous mask,
- per-thread evaluation of the bright / dark bit masks,
- warp–wide ballot_sync() to compact 32 corner flags into a single 32-bit word, then
- a single store by lane-0, greatly reducing global writes.

For a candidate pixel be the 16 circle samples and let

```math
B_i = \bigl[p_i \ge c + t \bigr], \qquad
D_i = \bigl[p_i \le c - t \bigr].
```

A FAST-9 corner is declared if

```math
\exists\, s\in[0,15]\;:\;
\Bigl(\bigwedge_{k=0}^{8} B_{(s+k)\,\bmod 16}\Bigr)\;\vee\;
\Bigl(\bigwedge_{k=0}^{8} D_{(s+k)\,\bmod 16}\Bigr).
```

That is, there is at least one arc of 9 consecutive samples that are all brighter than c+t or all darker than c-t.

- CUDA kernels:
    - fastKernel: warp-size (32-thread) kernel that pre-loads the 16 offsets and 9-bit masks from constant memory, builds the bright and dark 16-bit masks, checks them against the 16 pre-computed 9-bit masks, performs warp‐wide ballot to pack 32 flags into one 32-bit register, and writes the packed results to global memory from lane 0 only.
    - readPGM: minimal PGM (P5) reader for quick testing.
    
2) Implementation

Compiling the code:

<pre>nvcc .\fast.cu -o fast</pre>

Running the code after compiling:

<pre>fast</pre>

<pre>Random 4096×4096 image, threshold=20
CPU total               : 770.096 ms
GPU H2D copy            : 3.82858 ms
GPU kernel              : 0.84864 ms
GPU D2H copy            : 3.36912 ms
GPU total               : 15.7716 ms
Pixel disagreement      : 0 %</pre>