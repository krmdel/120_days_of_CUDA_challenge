Day 26: Implementation of logistic regression with mini-batch training and inference in CUDA

1) Summary of the daily tutorial:


2) Implementation:


Compiling the code:  

<pre>nvcc .\logisticregression.cu -o logistic_regression</pre>

Running the code after compiling: 
<pre> logistic_regression </pre>

- Sample size: 1000 / Feature size: 10  

<pre>
Starting CPU Training...
CPU Epoch 1/10 Loss: 0.228333 Epoch Time (CPU): 0.1213 ms
CPU Epoch 2/10 Loss: 0.228228 Epoch Time (CPU): 0.1032 ms
CPU Epoch 3/10 Loss: 0.228132 Epoch Time (CPU): 0.1029 ms
CPU Epoch 4/10 Loss: 0.228028 Epoch Time (CPU): 0.1072 ms
CPU Epoch 5/10 Loss: 0.227943 Epoch Time (CPU): 0.1066 ms
CPU Epoch 6/10 Loss: 0.227849 Epoch Time (CPU): 0.1357 ms
CPU Epoch 7/10 Loss: 0.227761 Epoch Time (CPU): 0.1048 ms
CPU Epoch 8/10 Loss: 0.227682 Epoch Time (CPU): 0.1057 ms
CPU Epoch 9/10 Loss: 0.227601 Epoch Time (CPU): 0.1062 ms
CPU Epoch 10/10 Loss: 0.227524 Epoch Time (CPU): 0.1052 ms
Total CPU Training Time: 1.0988 ms

CPU Inference Time: 0.0217 ms

Starting GPU Training...
GPU Epoch 1/10 Loss: 0.228334 Epoch Time (CPU): 2.7133 ms
GPU Epoch 2/10 Loss: 0.228225 Epoch Time (CPU): 1.7165 ms
GPU Epoch 3/10 Loss: 0.22813 Epoch Time (CPU): 1.7692 ms
GPU Epoch 4/10 Loss: 0.228026 Epoch Time (CPU): 1.8704 ms
GPU Epoch 5/10 Loss: 0.227942 Epoch Time (CPU): 1.6787 ms
GPU Epoch 6/10 Loss: 0.227849 Epoch Time (CPU): 1.6559 ms
GPU Epoch 7/10 Loss: 0.22776 Epoch Time (CPU): 1.7744 ms
GPU Epoch 8/10 Loss: 0.227677 Epoch Time (CPU): 2.0898 ms
GPU Epoch 9/10 Loss: 0.227602 Epoch Time (CPU): 2.2213 ms
GPU Epoch 10/10 Loss: 0.227527 Epoch Time (CPU): 1.9019 ms

Starting GPU Inference...

GPU Inference timings:
Host to Device: 0.025408 ms
Kernel Execution: 0.013312 ms
Device to Host: 0.093472 ms

CPU Inference Time (after GPU training): 0.0221 ms

Total GPU Training Timings:
Total Host to Device (training): 2.31341 ms
Total Kernel Execution (training): 3.66587 ms
Total Device to Host (training): 4.44749 ms</pre>

- Sample size: 10000 / Feature size: 1000  

<pre>Starting CPU Training...
CPU Epoch 1/10 Loss: -1.19209e-07 Epoch Time (CPU): 85.3623 ms
CPU Epoch 2/10 Loss: -1.19209e-07 Epoch Time (CPU): 86.7674 ms
CPU Epoch 3/10 Loss: -1.19209e-07 Epoch Time (CPU): 85.568 ms
CPU Epoch 4/10 Loss: -1.19209e-07 Epoch Time (CPU): 88.7118 ms
CPU Epoch 5/10 Loss: -1.19209e-07 Epoch Time (CPU): 89.6076 ms
CPU Epoch 6/10 Loss: -1.19209e-07 Epoch Time (CPU): 89.6302 ms
CPU Epoch 7/10 Loss: -1.19209e-07 Epoch Time (CPU): 96.7759 ms
CPU Epoch 8/10 Loss: -1.19209e-07 Epoch Time (CPU): 86.1275 ms
CPU Epoch 9/10 Loss: -1.19209e-07 Epoch Time (CPU): 87.046 ms
CPU Epoch 10/10 Loss: -1.19209e-07 Epoch Time (CPU): 85.1318 ms
Total CPU Training Time: 880.729 ms

CPU Inference Time: 22.1891 ms

Starting GPU Training...
GPU Epoch 1/10 Loss: -1.19209e-07 Epoch Time (CPU): 82.045 ms
GPU Epoch 2/10 Loss: -1.19209e-07 Epoch Time (CPU): 78.296 ms
GPU Epoch 3/10 Loss: -1.19209e-07 Epoch Time (CPU): 78.7194 ms
GPU Epoch 4/10 Loss: -1.19209e-07 Epoch Time (CPU): 65.8374 ms
GPU Epoch 5/10 Loss: -1.19209e-07 Epoch Time (CPU): 65.8091 ms
GPU Epoch 6/10 Loss: -1.19209e-07 Epoch Time (CPU): 63.8608 ms
GPU Epoch 7/10 Loss: -1.19209e-07 Epoch Time (CPU): 63.5442 ms
GPU Epoch 8/10 Loss: -1.19209e-07 Epoch Time (CPU): 62.1166 ms
GPU Epoch 9/10 Loss: -1.19209e-07 Epoch Time (CPU): 66.2997 ms
GPU Epoch 10/10 Loss: -1.19209e-07 Epoch Time (CPU): 64.9527 ms

Starting GPU Inference...

GPU Inference timings:
Host to Device: 3.5375 ms
Kernel Execution: 0.16224 ms
Device to Host: 0.069184 ms

CPU Inference Time (after GPU training): 20.1333 ms

Total GPU Training Timings:
Total Host to Device (training): 79.4109 ms
Total Kernel Execution (training): 308.877 ms
Total Device to Host (training): 40.8767 ms</pre>