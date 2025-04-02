Day 31: Implementation of skip connections and ResNet architecture for convolutional neural networks in CUDA

1) Summary of the daily tutorial:

The code performs a basic ResNet-style convolutional neural network with skip connections using CUDA. The goal is to demonstrate both CPU and GPU implementations for the complete training pipeline – including the forward pass (convolution, batch normalization, ReLU, and element-wise addition for the skip connection), loss computation (using mean squared error), gradient computation (simulating backpropagation with gradients computed as 2·output/size), and weight updates via gradient descent – as well as separate inference passes on CPU and GPU.

- CNN Architecture:
    - Input: A multi-channel image (32×32×3)
    - Initial Convolution (Projection): The input is projected into a higher-dimensional feature space (64 channels) using a 3×3 convolution with stride 1 and padding. This operation preserves the spatial dimensions of the input.
    - Residual Block:
        - Convolution 1: A 3×3 convolution (stride 1, same padding) is applied on the projected features.
        - Batch Normalization & ReLU: The output of the first convolution is normalized on a per-channel basis (using computed mean and variance with learnable parameters γ and β) and then passed through a ReLU activation.
        - Convolution 2: A second 3×3 convolution is applied on the activated features.
        - Batch Normalization: The output of the second convolution is normalized.
        - Skip Connection: The output of the initial projection (before entering the residual block) is added element-wise to the output of the second batch normalization.
        - ReLU: A ReLU activation is applied to the result of the skip connection to produce the final feature map.
    - Loss and Training: A simple mean squared error (MSE) loss is computed on the final output. Dummy gradients (computed as 2·output/size) simulate backpropagation, and weight updates are performed using a basic gradient descent step. Although the training loop only updates the convolution weights, all forward operations (including batch normalization and ReLU) are executed to demonstrate the complete pipeline.

- GPU Kernels:
    - conv_forward_res: Implements the forward pass of a convolution with same padding for the residual block.
    - conv_weight_grad_res: Computes the gradients for the convolution weights by iterating over the spatial dimensions and channels.
    - batch_norm_forward_gpu: Normalizes the features per channel on the GPU.
    - relu_forward_gpu: Applies the ReLU activation function.
    - elementwise_add_gpu: Performs element-wise addition for the skip connection.
    - compute_dummy_grad: Computes a dummy gradient (2×output/size) for simulating backpropagation.
    - update_weights: Updates the model weights using a simple gradient descent update.

2) Implementation:

Compiling the code:  

<pre>nvcc .\resnet.cu -o resnet</pre>

Running the code after compiling: 

<pre> resnet </pre>

- width, height, channels: 32, 32, 3  

<pre>Starting CPU Training for ResNet Block...
CPU Epoch 1 Loss: 0.814885
CPU Epoch 2 Loss: 0.813688
CPU Epoch 3 Loss: 0.812513
CPU Epoch 4 Loss: 0.811345
CPU Epoch 5 Loss: 0.810212
CPU Epoch 6 Loss: 0.809099
CPU Epoch 7 Loss: 0.808007
CPU Epoch 8 Loss: 0.806934
CPU Epoch 9 Loss: 0.805870
CPU Epoch 10 Loss: 0.804827
Total CPU Training Time: 7179.000000 ms
CPU Inference Time: 354.000000 ms

Starting GPU Training for ResNet Block...
GPU Epoch 1 Loss: 0.803788
GPU Epoch 2 Loss: 0.802775
GPU Epoch 3 Loss: 0.801782
GPU Epoch 4 Loss: 0.800800
GPU Epoch 5 Loss: 0.799835
GPU Epoch 6 Loss: 0.798881
GPU Epoch 7 Loss: 0.797950
GPU Epoch 8 Loss: 0.797031
GPU Epoch 9 Loss: 0.796120
GPU Epoch 10 Loss: 0.795218
Total GPU Training Time: 27.807232 ms
GPU Inference Time: 0.988288 ms</pre>