Day 24: Implementation of forward pass for linear regression in CUDA

1) Daily tutorial:

- The code performs the forward pass for linear regression in CUDA
- Forward pass of a linear regression model is implemented using CUDA:

    Parameters:
    X: Input matrix of shape (num_samples, num_features)
    w: Weight vector
    b: Bias (scalar)

    Returns:
    y_pred: Output predictions

    CUDA kernel (linear_forward): Each thread computes one prediction: y = Xw + b for a sample (idx to identify which sample each thread is processing), however, there is no ground truth label.

Note that CPU is faster than GPU for sample size 1000 and feature size 10. However, GPU is significantly faster than CPU for sample size 10000 and feature size 1000. No further optimization such as shared memory access etc. in CUDA is implemented yet.

2) Implementation:

The code performs forward pass for lineaer regression model on the GPU

Compiling the code:  

<pre>nvcc forwardpass.cu -o fpass</pre>

Running the code after compiling: 
<pre> fpass </pre>

- Sample size: 1000 / Feature size: 10  

<pre>First 10 Predictions Comparison:
Index   True y  GPU y   CPU y
0       3.20224 3.20224 3.20224
1       3.67197 3.67197 3.67197
2       3.11238 3.11238 3.11238
3       3.57921 3.57921 3.57921
4       3.41579 3.41579 3.41579
5       2.50251 2.50251 2.50251
6       3.96771 3.96771 3.96771
7       4.2795  4.2795  4.2795
8       3.24364 3.24364 3.24364
9       4.33685 4.33685 4.33685

GPU Timing (ms):
Host to Device:       0.04608 ms
Kernel execution:     0.6736 ms
Device to Host:       0.03872 ms
Total GPU time:       0.7584 ms

Total CPU time:       0.0168 ms</pre>

- Sample size: 10000 / Feature size: 1000  

<pre>First 10 Predictions Comparison:
Index   True y  GPU y   CPU y
0       242.871 242.871 242.871
1       255.594 255.594 255.594
2       250.812 250.812 250.812
3       244.777 244.777 244.777
4       252.882 252.882 252.882
5       248.689 248.689 248.689
6       250.591 250.591 250.591
7       254.514 254.514 254.514
8       259.005 259.005 259.005
9       248.271 248.271 248.271

GPU Timing (ms):
Host to Device:       3.35872 ms
Kernel execution:     1.3097 ms
Device to Host:       0.162176 ms
Total GPU time:       4.83059 ms

Total CPU time:       19.2113 ms</pre>