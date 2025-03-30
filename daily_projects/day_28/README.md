Day 28: Implementation of multi layer perceptron with forward and backward pass in CUDA

1) Summary of the daily tutorial:

The code performs multi-layer perceptron (MLP) with one hidden layer for binary classification. The hidden layer applies the sigmoid activation function followed by an output layer that also uses sigmoid to generate probabilities.

- Forward Pass: Each CUDA thread computes the hidden layer activations by performing a weighted sum of the input features, adding a bias, and applying the sigmoid function. The output layer then computes the final prediction.

- Backward Pass: After binary cross entropy loss, the gradients for both the hidden and output layers are computed via backpropagation. The kernel uses atomic operations to safely accumulate gradient contributions from multiple threads.

- Weight Update: Model parameters (weights and biases) are updated using gradient descent with a fixed learning rate.

- Mini-Batch Training: The training is performed in mini-batches. Both CPU and GPU implementations process data in batches. CUDA events measure individual timings for Host-to-Device transfers, kernel execution, and Device-to-Host transfers.

- Synthetic Data: Random synthetic data is generated using a “true” model for demonstration purposes. Although training effectiveness might be limited by the random nature of the data, the focus is on demonstrating the CUDA implementation.

2) Implementation:

Compiling the code:  

<pre>nvcc .\mlp.cu -o mlp</pre>

Running the code after compiling: 

<pre> mlp </pre>

- Sample size: 1000 / Feature size: 10  

<pre>Starting CPU Training for MLP...
CPU Epoch 1/10 Loss: 0.243699 Epoch Time (CPU): 0.3548 ms
CPU Epoch 2/10 Loss: 0.243451 Epoch Time (CPU): 0.5122 ms
CPU Epoch 3/10 Loss: 0.243207 Epoch Time (CPU): 0.3555 ms
CPU Epoch 4/10 Loss: 0.242951 Epoch Time (CPU): 0.356 ms
CPU Epoch 5/10 Loss: 0.242729 Epoch Time (CPU): 0.3555 ms
CPU Epoch 6/10 Loss: 0.242515 Epoch Time (CPU): 0.3559 ms
CPU Epoch 7/10 Loss: 0.242303 Epoch Time (CPU): 0.3562 ms
CPU Epoch 8/10 Loss: 0.242107 Epoch Time (CPU): 0.3526 ms
CPU Epoch 9/10 Loss: 0.241924 Epoch Time (CPU): 0.3652 ms
CPU Epoch 10/10 Loss: 0.241726 Epoch Time (CPU): 0.341 ms
Total CPU Training Time: 3.7049 ms

CPU Inference Time: 0.1431 ms

Starting GPU Training for MLP...
GPU Epoch 1/10 Loss: 0.242049 Epoch Time (CPU): 1.2984 ms
GPU Epoch 2/10 Loss: 0.24203 Epoch Time (CPU): 0.3458 ms
GPU Epoch 3/10 Loss: 0.242011 Epoch Time (CPU): 0.3619 ms
GPU Epoch 4/10 Loss: 0.241992 Epoch Time (CPU): 0.3756 ms
GPU Epoch 5/10 Loss: 0.241973 Epoch Time (CPU): 0.3285 ms
GPU Epoch 6/10 Loss: 0.241954 Epoch Time (CPU): 0.3501 ms
GPU Epoch 7/10 Loss: 0.241936 Epoch Time (CPU): 0.3675 ms
GPU Epoch 8/10 Loss: 0.241917 Epoch Time (CPU): 0.5241 ms
GPU Epoch 9/10 Loss: 0.241898 Epoch Time (CPU): 0.3786 ms
GPU Epoch 10/10 Loss: 0.24188 Epoch Time (CPU): 0.3043 ms

Total GPU Training Timings:
Total Host to Device (training): 0.347584 ms
Total Kernel Execution (training): 2.1712 ms
Total Device to Host (training): 0.466912 ms
Total GPU Training Time: 2.9857 ms

Starting GPU Inference for MLP...

GPU Inference Timings:
Host to Device: 0.132256 ms
Kernel Execution: 0.021504 ms
Device to Host: 0.035648 ms
Total GPU Inference Time: 0.189408 ms

CPU Inference Time (after GPU training): 0.1413 ms</pre>

- Sample size: 10000 / Feature size: 1000  

<pre>Starting CPU Training for MLP...
CPU Epoch 1/10 Loss: 0.0500312 Epoch Time (CPU): 328.277 ms
CPU Epoch 2/10 Loss: 0.0388112 Epoch Time (CPU): 335.884 ms
CPU Epoch 3/10 Loss: 0.0316611 Epoch Time (CPU): 335.466 ms
CPU Epoch 4/10 Loss: 0.0267156 Epoch Time (CPU): 334.158 ms
CPU Epoch 5/10 Loss: 0.0230954 Epoch Time (CPU): 335.76 ms
CPU Epoch 6/10 Loss: 0.0203327 Epoch Time (CPU): 335.218 ms
CPU Epoch 7/10 Loss: 0.0181563 Epoch Time (CPU): 329.94 ms
CPU Epoch 8/10 Loss: 0.0163981 Epoch Time (CPU): 334.278 ms
CPU Epoch 9/10 Loss: 0.0149484 Epoch Time (CPU): 351.67 ms
CPU Epoch 10/10 Loss: 0.013733 Epoch Time (CPU): 338.366 ms
Total CPU Training Time: 3359.02 ms

CPU Inference Time: 100.789 ms

Starting GPU Training for MLP...
GPU Epoch 1/10 Loss: 0.0168989 Epoch Time (CPU): 59.7095 ms
GPU Epoch 2/10 Loss: 0.0168989 Epoch Time (CPU): 60.8237 ms
GPU Epoch 3/10 Loss: 0.0168989 Epoch Time (CPU): 61.0318 ms
GPU Epoch 4/10 Loss: 0.0168989 Epoch Time (CPU): 59.1108 ms
GPU Epoch 5/10 Loss: 0.0168989 Epoch Time (CPU): 53.0974 ms
GPU Epoch 6/10 Loss: 0.0168989 Epoch Time (CPU): 63.3639 ms
GPU Epoch 7/10 Loss: 0.0168989 Epoch Time (CPU): 58.9727 ms
GPU Epoch 8/10 Loss: 0.0168989 Epoch Time (CPU): 49.2113 ms
GPU Epoch 9/10 Loss: 0.0168989 Epoch Time (CPU): 53.0241 ms
GPU Epoch 10/10 Loss: 0.0168989 Epoch Time (CPU): 56.6217 ms

Total GPU Training Timings:
Total Host to Device (training): 49.7471 ms
Total Kernel Execution (training): 244.278 ms
Total Device to Host (training): 8.26793 ms
Total GPU Training Time: 302.293 ms

Starting GPU Inference for MLP...

GPU Inference Timings:
Host to Device: 3.33581 ms
Kernel Execution: 1.05331 ms
Device to Host: 0.036992 ms
Total GPU Inference Time: 4.42611 ms

CPU Inference Time (after GPU training): 103.296 ms</pre>