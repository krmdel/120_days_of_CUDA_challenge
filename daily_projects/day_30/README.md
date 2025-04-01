Day 30: Implementation of pooling operations (max/average pooling), batch normalization and dropout for convolutional neural networks in CUDA

1) Summary of the daily tutorial:

The code performs a simple convolutional neural network (CNN) using CUDA. The goal is to demonstrate both CPU and GPU implementations for the forward pass (convolution), gradient computation (backward pass), and weight updates.

- CNN Architecture:
    - Input: A multi-channel image (e.g., 32×32×3 or 1024×1024×3)
    - Convolution: A single convolution layer is applied with a kernel size of 3×3 and a stride of 2.  
    OUT_HEIGHT = ((IN_HEIGHT - KERNEL_SIZE) / STRIDE + 1)  
    OUT_WIDTH = ((IN_WIDTH - KERNEL_SIZE) / STRIDE + 1)
    - Pooling: A pooling layer (max pooling or average pooling) is applied on the convolution output using a 2×2 window with a stride of 2, further reducing the spatial dimensions.  
    - Batch Normalization: Normalizes the pooled features on a per-channel basis using computed mean and variance with learnable parameters (γ and β) to stabilize training.- Dropout: Applies dropout regularization to the batch-normalized features by randomly zeroing out activations and scaling the remaining values to help prevent overfitting.
    - Final Output: The resulting feature map is used for inference. Note that the training loop only updates the convolution weights, while the new layers are demonstrated in the forward pass.

- GPU Kernels:
    - conv_forward: Implements the forward pass by computing the convolution over the input image using nested loops over channels, kernel rows, and columns
    - conv_weight_grad: Computes the gradients for the convolution weights by iterating over all positions where a kernel element is applied
    - update_weights: Updates the model parameters (weights) using a simple gradient descent step

2) Implementation:

Compiling the code:  

<pre>nvcc .\cnn_update.cu -o cnnupdate</pre>

Running the code after compiling: 

<pre> cnnupdate </pre>

- width, height, channels: 32, 32, 3  

<pre>Starting CPU Training for CNN (convolution only)...
CPU Epoch 1/10 Loss: 0.341948 Epoch Time (CPU): 0.0000 ms
CPU Epoch 2/10 Loss: 0.293666 Epoch Time (CPU): 0.0000 ms
CPU Epoch 3/10 Loss: 0.257535 Epoch Time (CPU): 0.0000 ms
CPU Epoch 4/10 Loss: 0.230467 Epoch Time (CPU): 0.0000 ms
CPU Epoch 5/10 Loss: 0.210160 Epoch Time (CPU): 0.0000 ms
CPU Epoch 6/10 Loss: 0.194895 Epoch Time (CPU): 0.0000 ms
CPU Epoch 7/10 Loss: 0.183391 Epoch Time (CPU): 0.0000 ms
CPU Epoch 8/10 Loss: 0.174694 Epoch Time (CPU): 0.0000 ms
CPU Epoch 9/10 Loss: 0.168089 Epoch Time (CPU): 0.0000 ms
CPU Epoch 10/10 Loss: 0.163045 Epoch Time (CPU): 0.0000 ms

Starting GPU Training for CNN (convolution only)...
GPU Epoch 1/10 Loss: 0.341948 Epoch Time (CPU): 1.0000 ms
GPU Epoch 2/10 Loss: 0.293666 Epoch Time (CPU): 1.0000 ms
GPU Epoch 3/10 Loss: 0.257535 Epoch Time (CPU): 0.0000 ms
GPU Epoch 4/10 Loss: 0.230467 Epoch Time (CPU): 1.0000 ms
GPU Epoch 5/10 Loss: 0.210160 Epoch Time (CPU): 0.0000 ms
GPU Epoch 6/10 Loss: 0.194895 Epoch Time (CPU): 1.0000 ms
GPU Epoch 7/10 Loss: 0.183391 Epoch Time (CPU): 0.0000 ms
GPU Epoch 8/10 Loss: 0.174694 Epoch Time (CPU): 0.0000 ms
GPU Epoch 9/10 Loss: 0.168089 Epoch Time (CPU): 1.0000 ms
GPU Epoch 10/10 Loss: 0.163046 Epoch Time (CPU): 0.0000 ms

--- Timing Results ---
CPU Training: Total Epoch Time (sum): 0.0000 ms
CPU Training: Wall Clock Time: 4.000000 ms
CPU Inference Time (including new layers): 0.0000 ms

--- GPU Training Timings (convolution only) ---
Total Host to Device (training): 0.544064 ms
Total Kernel Execution (training): 1.877152 ms
Total Device to Host (training): 0.500960 ms
Total GPU Training Time: 5.227648 ms

--- GPU Inference Timings ---
Host to Device: 0.019456 ms
Convolution Kernel Execution: 0.024576 ms
Pooling Kernel: 0.095232 ms
Batch Norm Kernel: 0.082944 ms
Dropout Kernel: 0.078848 ms
Device to Host: 0.033984 ms
Total GPU Inference Time (with new layers): 0.335040 ms</pre>

- width, height, channels: 1024, 1024, 3  

<pre>Starting CPU Training for CNN (convolution only)...
CPU Epoch 1/10 Loss: 0.196693 Epoch Time (CPU): 42.0000 ms
CPU Epoch 2/10 Loss: 0.195322 Epoch Time (CPU): 44.0000 ms
CPU Epoch 3/10 Loss: 0.194137 Epoch Time (CPU): 42.0000 ms
CPU Epoch 4/10 Loss: 0.193091 Epoch Time (CPU): 43.0000 ms
CPU Epoch 5/10 Loss: 0.192154 Epoch Time (CPU): 43.0000 ms
CPU Epoch 6/10 Loss: 0.191289 Epoch Time (CPU): 44.0000 ms
CPU Epoch 7/10 Loss: 0.190487 Epoch Time (CPU): 44.0000 ms
CPU Epoch 8/10 Loss: 0.189732 Epoch Time (CPU): 43.0000 ms
CPU Epoch 9/10 Loss: 0.189006 Epoch Time (CPU): 44.0000 ms
CPU Epoch 10/10 Loss: 0.188309 Epoch Time (CPU): 46.0000 ms

Starting GPU Training for CNN (convolution only)...
GPU Epoch 1/10 Loss: 0.196693 Epoch Time (CPU): 16.0000 ms
GPU Epoch 2/10 Loss: 0.195322 Epoch Time (CPU): 15.0000 ms
GPU Epoch 3/10 Loss: 0.194137 Epoch Time (CPU): 15.0000 ms
GPU Epoch 4/10 Loss: 0.193091 Epoch Time (CPU): 15.0000 ms
GPU Epoch 5/10 Loss: 0.192154 Epoch Time (CPU): 15.0000 ms
GPU Epoch 6/10 Loss: 0.191289 Epoch Time (CPU): 15.0000 ms
GPU Epoch 7/10 Loss: 0.190487 Epoch Time (CPU): 15.0000 ms
GPU Epoch 8/10 Loss: 0.189732 Epoch Time (CPU): 16.0000 ms
GPU Epoch 9/10 Loss: 0.189006 Epoch Time (CPU): 16.0000 ms
GPU Epoch 10/10 Loss: 0.188309 Epoch Time (CPU): 15.0000 ms

--- Timing Results ---
CPU Training: Total Epoch Time (sum): 435.0000 ms
CPU Training: Wall Clock Time: 439.000000 ms
CPU Inference Time (including new layers): 29.0000 ms

--- GPU Training Timings (convolution only) ---
Total Host to Device (training): 1.759136 ms
Total Kernel Execution (training): 143.243256 ms
Total Device to Host (training): 2.155616 ms
Total GPU Training Time: 156.744385 ms

--- GPU Inference Timings ---
Host to Device: 1.172128 ms
Convolution Kernel Execution: 0.085728 ms
Pooling Kernel: 0.043008 ms
Batch Norm Kernel: 5.570496 ms
Dropout Kernel: 0.130048 ms
Device to Host: 0.128032 ms
Total GPU Inference Time (with new layers): 7.129440 ms</pre>