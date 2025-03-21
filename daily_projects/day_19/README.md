Day 19: Intra Warp Synchronization

1) Resources:

The lecture 20 of GPU Computing by Dr. Izzat El Hajj  
Lecture 20: https://youtu.be/g5ZKBH6UQvE?si=GBAMHBaBrSwPPFxg

2) Topics Covered:

- Intr-warp synchronization

3) Summary of the Lecture:  

- Taking advantage of the special relationship between threads in the same warp to synchronize between them quickly.
- CUDA has several built-in functions to synchronize between warps: 
    - sharing (shuffling) data across threads
    - voting across threads
- Warp shuffle functions:
    - built-in warp shuffle functions enable threads to share data with other threads in the same warp: faster than using shared memory and __syncthreads() to share across threads in the same block
- Variants:
    - __shfl_sync(): direct copy from indexed lane
    - __shfl_up_sync(): copy from a lane with lower ID relative to caller
    - __shfl_down_sync(): copy from a lane with higher ID relative to caller
    - __shfl_xor_sync(): copy from a lane based on bitwise XOR of own lane ID

- Warp vote functions:
    -Built-in warp vote functions enable threads to vote on which threads in the warp satusfy some condition
    - Variants:
        - __all_sync(): check if predicate is nonzero of all threads
        - __any_sync(): check if predicate is nonzero for some thread
        - __ballot_sync(): return mask of which threads evaluate predicate to nonzero
        - __activemask(): return mask of which threads are active

- Optimization with warp vote:
    - Optimization: have only one thread in the warp increment the counter on behalf of the others
    - Steps:
        - Assing a leader thread: 
            - find which threads are active using __activemask()
                if threads 1,2,4,5 are active:
                thread index: 7 6 5 4 3 2 1 0
                active mask:  0 0 1 1 0 1 1 0  
            - Designate the first active thread as the leader
                - Find the first set bit in the active mask
                - Use __ffs(mask):
                    - returns index of the first set bit in a mask
                    - index starts from 1
                        - return value of 0 means no bits are set  
        - Find how many threads need to add to the queue
            - Count the number of set bits in the active mask
                - Use __popc(mask):
                    - returns the number of active threads
        - Have the leader perform the atomic operation
        - Broadcast the result to the other threads
            - Use: __shfl_sync()
        - Find offset of each active thread and store the result
            thread index: 7 6 5 4 3 2 1 0
            active mask:  0 0 1 1 0 1 1 0  
            offset:       - - 3 2 - 1 0 -
            - equivalent to finding the number of previous threads that are active
            thread index: 7 6 5 4 3 2 1 0
            active mask:  0 0 1 1 0 1 1 0
                                &  
            pre. threads: 0 0 0 0 1 1 1 1 --> shift 1 by thread index then subtract 1
                                =
            p.act.threads:0 0 0 0 0 1 1 0 --> use __popc() to count the number of set bits

4) Implementation:

The code performs Sparse Matrix-Vector Multiplication (SpMV) for ELLPACK (ELL) and Jagged Diagonal Storage (JDS) on the GPU

Compiling the code:  

<pre>nvcc .\spmv_ell_jds.cu -o elljds</pre>

Running the code after compiling: 
<pre> elljds </pre>

<pre>ELL copy time: 0.039936 ms
ELL kernel time: 0.831488 ms
ELL copy back time: 0.038912 ms
ELL total time: 0.910336 ms

JDS copy time: 0.096256 ms
JDS kernel time: 0.055296 ms
JDS copy back time: 0.054848 ms
JDS total time: 0.206400 ms</pre>