Day 29: Implementation of convolution neural networks in CUDA

1) Summary of the daily tutorial:

The code performs a simple convolutional neural network (CNN) using CUDA. The goal is to demonstrate both CPU and GPU implementations for the forward pass (convolution), gradient computation (backward pass), and weight updates.

- CNN Architecture: 
    - Input: A multi-channel image (e.g., 32×32×3)
    - Convolution: A single convolution layer is applied with a kernel size of 3×3 and a stride of 2
    - Output: The result is a downsampled feature map with dimensions computed as:
        OUT_HEIGHT = ((IN_HEIGHT - KERNEL_SIZE) / STRIDE + 1)
        OUT_WIDTH = ((IN_WIDTH - KERNEL_SIZE) / STRIDE + 1)

- GPU Kernels:
    - conv_forward: Implements the forward pass by computing the convolution over the input image using nested loops over channels, kernel rows, and columns
    - conv_weight_grad: Computes the gradients for the convolution weights by iterating over all positions where a kernel element is applied
    - update_weights: Updates the model parameters (weights) using a simple gradient descent step

2) Implementation:

Compiling the code:  

<pre>nvcc .\cnn.cu -o cnn</pre>

Running the code after compiling: 

<pre> cnn </pre>

- width, height, channels: 32, 32, 3  

<pre>Starting CPU Training for CNN...
CPU Epoch 1/10 Loss: 0.577421 Epoch Time (CPU): 0.0000 ms
CPU Epoch 2/10 Loss: 0.480040 Epoch Time (CPU): 0.0000 ms
CPU Epoch 3/10 Loss: 0.407655 Epoch Time (CPU): 0.0000 ms
CPU Epoch 4/10 Loss: 0.353801 Epoch Time (CPU): 0.0000 ms
CPU Epoch 5/10 Loss: 0.313684 Epoch Time (CPU): 0.0000 ms
CPU Epoch 6/10 Loss: 0.283750 Epoch Time (CPU): 0.0000 ms
CPU Epoch 7/10 Loss: 0.261366 Epoch Time (CPU): 1.0000 ms
CPU Epoch 8/10 Loss: 0.244579 Epoch Time (CPU): 0.0000 ms
CPU Epoch 9/10 Loss: 0.231942 Epoch Time (CPU): 0.0000 ms
CPU Epoch 10/10 Loss: 0.222382 Epoch Time (CPU): 0.0000 ms

Starting GPU Training for CNN...
GPU Epoch 1/10 Loss: 0.577421 Epoch Time (CPU): 2.0000 ms
GPU Epoch 2/10 Loss: 0.480040 Epoch Time (CPU): 1.0000 ms
GPU Epoch 3/10 Loss: 0.407656 Epoch Time (CPU): 0.0000 ms
GPU Epoch 4/10 Loss: 0.353801 Epoch Time (CPU): 1.0000 ms
GPU Epoch 5/10 Loss: 0.313684 Epoch Time (CPU): 0.0000 ms
GPU Epoch 6/10 Loss: 0.283750 Epoch Time (CPU): 0.0000 ms
GPU Epoch 7/10 Loss: 0.261366 Epoch Time (CPU): 0.0000 ms
GPU Epoch 8/10 Loss: 0.244579 Epoch Time (CPU): 1.0000 ms
GPU Epoch 9/10 Loss: 0.231942 Epoch Time (CPU): 0.0000 ms
GPU Epoch 10/10 Loss: 0.222382 Epoch Time (CPU): 0.0000 ms

--- Timing Results ---
CPU Training: Total Epoch Time (sum): 1.0000 ms
CPU Training: Wall Clock Time: 1.000000 ms
CPU Inference Time: 0.0000 ms

--- GPU Training Timings ---
Total Host to Device (training): 0.308416 ms
Total Kernel Execution (training): 2.430112 ms
Total Device to Host (training): 0.404064 ms
Total GPU Training Time: 5.319680 ms

--- GPU Inference Timings ---
Host to Device: 0.067584 ms
Kernel Execution: 0.033792 ms
Device to Host: 0.067328 ms
Total GPU Inference Time: 0.168704 ms</pre>

- width, height, channels: 1024, 1024, 3  

<pre>Starting CPU Training for CNN...
CPU Epoch 1/10 Loss: 0.464733 Epoch Time (CPU): 41.0000 ms
CPU Epoch 2/10 Loss: 0.390771 Epoch Time (CPU): 42.0000 ms
CPU Epoch 3/10 Loss: 0.335502 Epoch Time (CPU): 44.0000 ms
CPU Epoch 4/10 Loss: 0.294163 Epoch Time (CPU): 44.0000 ms
CPU Epoch 5/10 Loss: 0.263220 Epoch Time (CPU): 43.0000 ms
CPU Epoch 6/10 Loss: 0.240004 Epoch Time (CPU): 42.0000 ms
CPU Epoch 7/10 Loss: 0.222565 Epoch Time (CPU): 42.0000 ms
CPU Epoch 8/10 Loss: 0.209418 Epoch Time (CPU): 43.0000 ms
CPU Epoch 9/10 Loss: 0.199479 Epoch Time (CPU): 42.0000 ms
CPU Epoch 10/10 Loss: 0.191923 Epoch Time (CPU): 44.0000 ms

Starting GPU Training for CNN...
GPU Epoch 1/10 Loss: 0.464733 Epoch Time (CPU): 19.0000 ms
GPU Epoch 2/10 Loss: 0.390771 Epoch Time (CPU): 19.0000 ms
GPU Epoch 3/10 Loss: 0.335502 Epoch Time (CPU): 19.0000 ms
GPU Epoch 4/10 Loss: 0.294163 Epoch Time (CPU): 19.0000 ms
GPU Epoch 5/10 Loss: 0.263220 Epoch Time (CPU): 20.0000 ms
GPU Epoch 6/10 Loss: 0.240004 Epoch Time (CPU): 20.0000 ms
GPU Epoch 7/10 Loss: 0.222565 Epoch Time (CPU): 19.0000 ms
GPU Epoch 8/10 Loss: 0.209418 Epoch Time (CPU): 20.0000 ms
GPU Epoch 9/10 Loss: 0.199479 Epoch Time (CPU): 19.0000 ms
GPU Epoch 10/10 Loss: 0.191923 Epoch Time (CPU): 19.0000 ms

--- Timing Results ---
CPU Training: Total Epoch Time (sum): 427.0000 ms
CPU Training: Wall Clock Time: 432.000000 ms
CPU Inference Time: 26.0000 ms

--- GPU Training Timings ---
Total Host to Device (training): 1.743520 ms
Total Kernel Execution (training): 181.233429 ms
Total Device to Host (training): 2.255616 ms
Total GPU Training Time: 194.964478 ms

--- GPU Inference Timings ---
Host to Device: 1.420352 ms
Kernel Execution: 0.264672 ms
Device to Host: 0.197888 ms
Total GPU Inference Time: 1.882912 ms</pre>