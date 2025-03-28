Day 26: Implementation of backward pass with gradient calculations for updating weights of linear regression in CUDA

1) Summary of the daily tutorial:

- The code performs one training step for a linear regression model (including forward and backward pass with updating weights) in CUDA

- Additional backward pass and weight updates of a linear regression model:
    Parameters:
    X: Input matrix of shape (num_samples, num_features)
    w: Weight vector
    b: Bias (scalar)

    Returns:
    y_true: "True" labels (computed as y = Xw+b from the initial random parameters)
    y_pred: Output predictions

- A backward pass: Computing the gradients of the loss (using mean squared error error) with respect to the weights and bias  
- Weight update: Updating the weights and bias using gradient descent  

linear_forward: Each thread computes one prediction  
compute_gradients: Each thread computes the error between the prediction and the true value, and then atomically adds its contribution to the gradients  
update_weights: Updates the weights and bias using the computed gradients and a learning rate  

Note that CPU is faster than GPU for sample size 1000 and feature size 10. However, GPU is significantly faster than CPU for sample size 10000 and feature size 1000. No further optimization such as shared memory access etc. in CUDA is implemented yet.

2) Implementation:

The code performs backward pass with gradient calculations for updating weights for lineaer regression model on the GPU

Compiling the code:  

<pre>nvcc backpass.cu -o bpass</pre>

Running the code after compiling: 
<pre> bpass </pre>

- Sample size: 1000 / Feature size: 10  

<pre>First 10 Predictions (after 1 update):
Idx     True y  GPU y   CPU y
0       3.5585  3.5585  3.5585
1       3.59063 3.59063 3.59063
2       3.17322 3.17322 3.17322
3       3.57937 3.57937 3.57937
4       2.36742 2.36742 2.36742
5       3.77582 3.77582 3.77582
6       3.37447 3.37447 3.37447
7       4.92454 4.92454 4.92454
8       3.30885 3.30885 3.30885
9       2.98267 2.98267 2.98267

--- GPU Timing (ms) ---
Copy Host to Device: 0.055264 ms
Kernel Execution:    0.997344 ms
Copy Device to Host: 0.080768 ms
Total GPU time:      1.13338 ms

--- CPU Timing ---
Total CPU time:      0.0741 ms</pre>

- Sample size: 10000 / Feature size: 1000  

<pre>First 10 Predictions (after 1 update):
Idx     True y  GPU y   CPU y
0       257.053 257.053 257.053
1       249.518 249.518 249.518
2       253.809 253.809 253.809
3       257.901 257.901 257.901
4       246.914 246.914 246.914
5       258.842 258.842 258.842
6       247.999 247.998 247.999
7       249.273 249.273 249.273
8       246.976 246.976 246.976
9       257.535 257.535 257.535

--- GPU Timing (ms) ---
Copy Host to Device: 3.7345 ms
Kernel Execution:    8.57578 ms
Copy Device to Host: 0.100928 ms
Total GPU time:      12.4112 ms

--- CPU Timing ---
Total CPU time:      60.087 ms</pre>