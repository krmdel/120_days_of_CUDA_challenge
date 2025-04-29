Day 58: Implementation of StyleGAN-2 in CUDA

1) Summary of the daily tutorial

The code implements the core components of a StyleGAN-2 generator in CUDA, stitching together a mapping network that transforms a random latent z into style vectors w, and a sequence of modulated convolution blocks that synthesize feature maps at increasing resolutions. You’ll see how to launch lightweight kernels for fully connected layers, leaky-ReLU activations, bias+nonlinearity, and a custom 3×3 "modulated" convolution that applies per-sample style scaling to the convolution weights.

```math
w^{(l)}_n = \mathrm{LReLU}\bigl(W^{(l)}\,w^{(l-1)}_n + b^{(l)}\bigr)
```  
for each layer \(l\) of the mapping network, and  
```math
y_{n,co,h,w} = \sum_{ci=0}^{C_{in}-1} \bigl(s_{n,ci}\times w^{3\times3}_{co,ci}\bigr)\;\ast\;x_{n,ci}\;+\;b_{co}
```  
for the modulated-convolution block, where \(s_{n,ci}\) is the style coefficient for batch sample \(n\), channel \(ci\).

- CUDA kernels:
  - fill_const: Fills the first n elements of p with constant v, used to zero- or bias-initialize tensors.  
  - lrelu_inplace: Applies the leaky-ReLU activation x --> max(x,0.2x) in place on an array of length n.  
  - bias_act: Adds a per-channel bias and then applies tanh nonlinearity to each feature map of size CxHxW.  
  - mod_conv3x3: Performs a 3×3 convolution where the weights are dynamically modulated by per-sample style style[n*Cin+ci], to produce output feature map y.  
  - fc512: Implements a fully-connected layer mapping 512-dim inputs to 512-dim outputs for each of the N samples in the batch.

2) Implementation

Compiling the code:

<pre>nvcc .\stylegan2.cu -o stylegan2</pre>

Running the code after compiling:

<pre>stylegan2</pre>

<pre>Micro-StyleGAN2 | 16x16 | B=16 | steps=20 | LR_G 2.500e-03 LR_D 2.500e-03
Iter   0 | H2D 0.022 ms | Kern 44.446 ms | D2H 0.013 ms | Tot 44.481 ms
Iter   1 | H2D 0.022 ms | Kern 0.591 ms | D2H 0.002 ms | Tot 0.615 ms
Iter   2 | H2D 0.047 ms | Kern 0.653 ms | D2H 0.002 ms | Tot 0.702 ms
Iter   3 | H2D 0.045 ms | Kern 0.614 ms | D2H 0.002 ms | Tot 0.660 ms
Iter   4 | H2D 0.080 ms | Kern 0.665 ms | D2H 0.002 ms | Tot 0.746 ms
Iter   5 | H2D 0.060 ms | Kern 0.681 ms | D2H 0.002 ms | Tot 0.743 ms
Iter   6 | H2D 0.090 ms | Kern 0.668 ms | D2H 0.002 ms | Tot 0.759 ms
Iter   7 | H2D 0.022 ms | Kern 0.592 ms | D2H 0.002 ms | Tot 0.615 ms
Iter   8 | H2D 0.046 ms | Kern 0.636 ms | D2H 0.002 ms | Tot 0.684 ms
Iter   9 | H2D 0.044 ms | Kern 0.631 ms | D2H 0.002 ms | Tot 0.677 ms
Iter  10 | H2D 0.059 ms | Kern 0.633 ms | D2H 0.002 ms | Tot 0.693 ms
Iter  11 | H2D 0.059 ms | Kern 0.634 ms | D2H 0.002 ms | Tot 0.694 ms
Iter  12 | H2D 0.022 ms | Kern 0.635 ms | D2H 0.002 ms | Tot 0.658 ms
Iter  13 | H2D 0.327 ms | Kern 0.594 ms | D2H 0.002 ms | Tot 0.923 ms
Iter  14 | H2D 0.225 ms | Kern 0.605 ms | D2H 0.002 ms | Tot 0.832 ms
Iter  15 | H2D 0.059 ms | Kern 0.637 ms | D2H 0.002 ms | Tot 0.698 ms
Iter  16 | H2D 0.060 ms | Kern 0.704 ms | D2H 0.002 ms | Tot 0.765 ms
Iter  17 | H2D 0.021 ms | Kern 0.584 ms | D2H 0.002 ms | Tot 0.607 ms
Iter  18 | H2D 0.045 ms | Kern 1.461 ms | D2H 0.002 ms | Tot 1.508 ms
Iter  19 | H2D 0.045 ms | Kern 0.631 ms | D2H 0.002 ms | Tot 0.678 ms</pre>