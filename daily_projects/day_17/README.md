Day 17: Graph Processing

1) Resources:

The lecture 18 of GPU Computing by Dr. Izzat El Hajj  
Lecture 18: https://youtu.be/P3eQWkVj9dA?si=GnGNhUej1cIqM4h4

2) Topics Covered:

- Graph processing

3) Summary of the Lecture:  

- Graphs can be represented with adjacency matrices --> can use the same storage formats for sparse matrices
    - Unweighted graphs: 0s and 1s
    - Undirected graphs: symmetric matrices

- COO or CSR/CSC formats can be used for adjacency matrices. For unweighted and undirected graphs:
    - COO: no need to store values because graph is unweighted
    - CSR/CSC: they are equivalent because matrix is symmetric  

Parallelization approach:
    - Vertix-centric: assign a thread to each vertex.
        - typically use CSR/CSC format: Given a vertex, easy to find all its neighbors. might also use other formats (e.g., ELL, JDS) as optimization
    - Edge-centric: assign a thread to each edge
        - typically use COO format: Given an edge, easy to find its source and destination vertices.  
    - Hybrid:
        - For example: given an edge, need neighbors of source and destination vertices.
        - Use both COO and CSR/CSC formats

- Breadth-first search (BFS): find the distance (level) of each vertex from some source vertex
    - Approaches to parallelizing BFS:
        - Vertex-centric (two versions)
            - Top-down: For every vertex, if it was in the previous level, add all its unvisited neighbors to the current level (i.e., assign a thread to every parent vertex in BFS tree)
            - Bottom-up: For every vertex, if it has not been visited, if any of its neighbors are in the previous level, add it to the current level (i.e., assign a thread to every child vertex in BFS tree)
            - Direction-optimized: Start with top-down then switch to bottom-up: Bottom-up is inefficient at the beginning because most vertices will search all neighbors and not find any that are in the previous level
        - Edge-centric: For every edge, if its source vertex was in the previous level, add its destination vertex to the current level

- Dataset Implications:
    - The best parallelization approach depends on the structure of the graph
    - The vertex-centric bottom-up adnd the edge-centric approaches are better on high-degree graphs (e.g., social network graphs with celebrities), better at dealing with load imbalance
    - The vertex-centric top-down approach is better on low-degree graphs (e.g., road networks/map) --> can be improved by launching more threads to process neighbors of high-degree vertices

- Similarity between BFS and SpMV:
    - SpMV:  
        for every row
            for every nonzero in the row
                lookup in vector-in at column
                update the vector-out at row

    - BFS (vertex-centric bottom-up):  
        for every vertex
            for every edge of the vertex
                lookup in levels-old at neighbor
                update levels-new at vertex
        - Vertex-centric top-down and edge-centric also correspond to other ways of performing SpMV

- Linear Algebraic Formulation of Graph Problems:
    - With a few tweaks, BFS can be formulated exactly as SpMV: use visited list as input/output vector. A few other vector operations to get exactly what we want, but SpMV remains the dominant computation.
    - Many graph problems can be formulated in terms of sparse linear algebra computations
        - Advantage: leverage mature and well-optimized parallel libraries for high performance sparse linear algebra
        - Disadvantage: not always the most efficient way to solve the problem

- Redundant work:
    - Approaches discussed so far check every vertex or edge on every iteration for relevance
    - Strengths: easy to implement, high parallel, no synchronization across threads
    - Weaknesses: a lot of unnecessary threads/work: many threads will find that their vertex or edge are not relevant for this iteration and just exit

- Work-efficient approach that avoids launching threads for irrelevant vertices or edges, but requires more synchronization across threads (lecture 19)

4) Implementation:

The code performs sort using radix and merge sorting algorithms and compare 1 and 2 digit sizes with global and shared memory accesses on the GPU

Compiling the code:  

<pre>nvcc .\sort.cu -o sort</pre>

Running the code after compiling: 
<pre> sort </pre>

<pre>Running Radix Sort Baseline and Optimized:
Comparing least significant digit (width=1) vs two digits (width=2):
Radix Sort (digit width: 1, optimized: 0) kernel time: 0.579584 ms
Radix Sort (digit width: 1, optimized: 1) kernel time: 0.010240 ms
Radix Sort (digit width: 2, optimized: 0) kernel time: 0.006144 ms
Radix Sort (digit width: 2, optimized: 1) kernel time: 0.006144 ms
Running Merge Sort Baseline and Optimized:
Merge Sort (optimized: 0) total kernel time: 0.284672 ms
Merge Sort (optimized: 1) total kernel time: 0.291840 ms</pre>
