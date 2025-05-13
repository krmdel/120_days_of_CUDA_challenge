Day 71: Implementation of one dimensional (1D) circular convolution using cuFFT

1) Summary of the daily tutorial

The code implements a circular convolution between two real-valued 1-D signals by leveraging cuFFT for FFT / IFFT operations and a tiny custom kernel for the element-wise complex multiplication in the frequency domain.

Circular convolution of length-N signals x[n] and h[n] is defined as

    ```math
    y[n] \;=\; \sum_{k=0}^{N-1} x[k]\,h\!\bigl[(n-k)\bmod N\bigr],
    \qquad n = 0,\dots,N-1
    ```

	- Forward FFT of both signals
    ```math
    X[k] \;=\; \text{FFT}\{x[n]\}, \qquad H[k] \;=\; \text{FFT}\{h[n]\}	
    ```

    - Point-wise complex product in the frequency domain
    ```math
    Y[k] = X[k] \cdot H[k]
    ```

    - Inverse FFT and normalisation
    ```math
    y[n] = \tfrac{1}{N}\,\text{IFFT}\{Y[k]\}
    ```

    - CUDA kernels:
	    complexPointwiseMul: multiplies two complex vectors element-wise:

            a[i] = cuCmulf(a[i], b[i]) --> Y[k] = X[k] · H[k]

        After the kernel completes, d_x contains Y[k] and is reused for the inverse FFT.

        - Reads one element of each input spectrum.
        - Performs one complex multiply (cuCmulf).
        - Writes the result back to global memory in-place.

2) Implementation

Compiling the code:

<pre>nvcc .\1d_circular_cufft.cu -o 1d_circular_cufft</pre>

Running the code after compiling:

<pre>1d_circular_cufftd</pre>

<pre>Vector length: 16384

Inference time: 0.10752 ms</pre>