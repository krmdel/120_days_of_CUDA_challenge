Day 57: Implementation of deep convolutional generative adversarial network (DCGAN) in CUDA

1) Summary of the daily tutorial
  The code implements and train a simple GAN in CUDA, using fully-connected ("deep convolutional") layers to map 100-dimensional noise vectors to 28×28 images and back. The host code launches waves of CUDA kernels to perform the forward passes for both generator and discriminator, compute binary-cross-entropy losses and gradients, and finally update all weights via SGD.  

  Throughout training, the standard GAN minimax objective is optimized:  
  ```math
  V(D, G) \;=\; \mathbb{E}_{x\sim p_{\mathrm{data}}}\bigl[\log D(x)\bigr]
             + \mathbb{E}_{z\sim p_z}\bigl[\log\bigl(1 - D(G(z))\bigr)\bigr]\,.  
  ```  

  - CUDA kernels:  
    - fill: launches enough threads to set every element of a float array to a constant.  
    - relu / relu_g: element-wise ReLU and its derivative (for backprop).  
    - tanh_f / tanh_g: element-wise tanh and its derivative.  
    - sigmoid: transforms logits into probabilities in [0,1].  
    - bce_grad: computes the binary-cross-entropy loss per element, its gradient, and reduces into a single loss scalar via shared-memory tree reduction + 'atomicAdd'.  
    - lin_fwd: computes a dense layer forward pass (matrix–vector multiply plus bias) over a batch.  
    - dW_db: from upstream dY, accumulates weight‐gradient dW via Xᵀ·dY and bias‐gradient db by summing dY.  
    - **dX: given dY and W, computes input-gradient dX = dY·Wᵀ.  
    - **sgd: performs the update  

    ```math
      w_i \;\leftarrow\; w_i - \tfrac{\mathrm{LR}}{\mathrm{BATCH}}\,\bigl.\tfrac{\partial\mathcal{L}}{\partial w_i}\bigr.
    ```  
    for every element of a weight or bias tensor.  

2) Implementation

Compiling the code:

<pre>nvcc .\dcgan.cu -o dcgan</pre>

Running the code after compiling:

<pre>dcgan</pre>

<pre>Running DCGAN (FC) | LR 2.0e-04
Epoch   0 | H2D 0.066 ms | Kern 1.831 ms | D2H 0.094 ms | Tot 1.991 ms | D -0.0002 | G 0.0003
Epoch   1 | H2D 0.099 ms | Kern 0.887 ms | D2H 0.116 ms | Tot 1.102 ms | D -0.0000 | G 0.0006
Epoch   2 | H2D 0.102 ms | Kern 0.843 ms | D2H 0.119 ms | Tot 1.065 ms | D -0.0001 | G 0.0002
Epoch   3 | H2D 0.098 ms | Kern 0.846 ms | D2H 0.121 ms | Tot 1.065 ms | D -0.0002 | G 0.0005
Epoch   4 | H2D 0.100 ms | Kern 0.855 ms | D2H 0.107 ms | Tot 1.062 ms | D -0.0003 | G 0.0004
Epoch   5 | H2D 0.115 ms | Kern 0.842 ms | D2H 0.123 ms | Tot 1.080 ms | D -0.0004 | G 0.0007
Epoch   6 | H2D 0.064 ms | Kern 1.253 ms | D2H 0.073 ms | Tot 1.389 ms | D -0.0002 | G 0.0008
Epoch   7 | H2D 0.234 ms | Kern 0.878 ms | D2H 0.119 ms | Tot 1.231 ms | D -0.0000 | G 0.0006
Epoch   8 | H2D 0.107 ms | Kern 0.908 ms | D2H 0.122 ms | Tot 1.137 ms | D -0.0005 | G 0.0006
Epoch   9 | H2D 0.104 ms | Kern 0.842 ms | D2H 0.124 ms | Tot 1.070 ms | D -0.0004 | G 0.0004
Epoch  10 | H2D 0.071 ms | Kern 0.846 ms | D2H 0.060 ms | Tot 0.977 ms | D 0.0001 | G 0.0005
Epoch  11 | H2D 0.220 ms | Kern 0.933 ms | D2H 0.073 ms | Tot 1.226 ms | D 0.0002 | G 0.0002
Epoch  12 | H2D 0.118 ms | Kern 0.842 ms | D2H 0.123 ms | Tot 1.082 ms | D -0.0002 | G 0.0002
Epoch  13 | H2D 0.106 ms | Kern 0.846 ms | D2H 0.094 ms | Tot 1.046 ms | D -0.0001 | G 0.0003
Epoch  14 | H2D 0.110 ms | Kern 0.846 ms | D2H 0.124 ms | Tot 1.080 ms | D -0.0002 | G 0.0008
Epoch  15 | H2D 0.106 ms | Kern 0.845 ms | D2H 0.103 ms | Tot 1.054 ms | D -0.0001 | G 0.0004
Epoch  16 | H2D 0.258 ms | Kern 1.068 ms | D2H 0.121 ms | Tot 1.447 ms | D -0.0003 | G 0.0006
Epoch  17 | H2D 0.105 ms | Kern 0.845 ms | D2H 0.124 ms | Tot 1.074 ms | D -0.0005 | G 0.0001
Epoch  18 | H2D 0.075 ms | Kern 0.982 ms | D2H 0.124 ms | Tot 1.180 ms | D -0.0002 | G -0.0000
Epoch  19 | H2D 0.117 ms | Kern 0.845 ms | D2H 0.122 ms | Tot 1.083 ms | D 0.0002 | G 0.0003</pre>