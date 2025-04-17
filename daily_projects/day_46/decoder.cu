#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdlib>

// Error‐checking macro
#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err)               \
                << " in " << __FILE__ << " at line " << __LINE__ << "\n";  \
      std::exit(EXIT_FAILURE);                                             \
    }                                                                      \
  } while (0)

// Model parameters
#define D_MODEL       256
#define H_HEADS        8
#define D_HEAD        (D_MODEL / H_HEADS)
#define D_FF          (4 * D_MODEL)
#define MAX_TEXT_LEN  16
#define VOCAB_SIZE   10000
#define TILE_WIDTH    16
#define CROSS_M       64   // number of visual tokens

// CPU functions

// Matrix multiplication: C[M×N] = A[M×K] × B[K×N]
void cpu_matmul(const float* A, const float* B, float* C, int M, int K, int N) {
  for(int i = 0; i < M; i++) {
    for(int j = 0; j < N; j++) {
      float sum = 0.f;
      for(int k = 0; k < K; k++) {
        sum += A[i*K + k] * B[k*N + j];
      }
      C[i*N + j] = sum;
    }
  }
}

// Row‑wise softmax
void cpu_softmax(float* mat, int rows, int cols) {
  for(int r = 0; r < rows; r++) {
    float* row = mat + r*cols;
    float m = row[0], s = 0.f;
    for(int j = 1; j < cols; j++) if(row[j] > m) m = row[j];
    for(int j = 0; j < cols; j++) {
      row[j] = expf(row[j] - m);
      s += row[j];
    }
    for(int j = 0; j < cols; j++) row[j] /= s;
  }
}

// Single greedy decoder step on CPU
int cpu_decoder_step(
    const std::vector<float>& cross_feats,
    const std::vector<int>& ids,
    const float* token_emb,
    const float* Wq_s, const float* Wk_s, const float* Wv_s, const float* Wo_s,
    const float* Wq_x, const float* Wk_x, const float* Wv_x, const float* Wo_x,
    const float* W1,   const float* W2,    const float* Wout
) {
  int T = ids.size();
  float sf = 1.f / sqrtf((float)D_HEAD);

  // Embed tokens
  std::vector<float> X(T*D_MODEL);
  for(int i = 0; i < T; i++)
    for(int d = 0; d < D_MODEL; d++)
      X[i*D_MODEL + d] = token_emb[ ids[i]*D_MODEL + d ];

  // Self‑attention
  std::vector<float> Q(T*D_MODEL), K(T*D_MODEL), V(T*D_MODEL);
  cpu_matmul(X.data(), Wq_s, Q.data(), T, D_MODEL, D_MODEL);
  cpu_matmul(X.data(), Wk_s, K.data(), T, D_MODEL, D_MODEL);
  cpu_matmul(X.data(), Wv_s, V.data(), T, D_MODEL, D_MODEL);

  std::vector<float> KT(D_MODEL*T), scores(T*T), attn(T*D_MODEL), self_out(T*D_MODEL);
  for(int i = 0; i < T; i++)
    for(int d = 0; d < D_MODEL; d++)
      KT[d*T + i] = K[i*D_MODEL + d];

  cpu_matmul(Q.data(), KT.data(), scores.data(), T, D_MODEL, T);
  for(int i = 0; i < T*T; i++) scores[i] *= sf;
  cpu_softmax(scores.data(), T, T);
  cpu_matmul(scores.data(), V.data(), attn.data(), T, T, D_MODEL);
  cpu_matmul(attn.data(), Wo_s, self_out.data(), T, D_MODEL, D_MODEL);
  for(int i = 0; i < T*D_MODEL; i++) X[i] += self_out[i];

  // Cross‑attention for last token
  std::vector<float> Qc(D_MODEL),
                     Kc(CROSS_M*D_MODEL),
                     Vc(CROSS_M*D_MODEL),
                     KTc(D_MODEL*CROSS_M),
                     sc(CROSS_M),
                     ctxt(D_MODEL);
  for(int d = 0; d < D_MODEL; d++)
    Qc[d] = X[(T-1)*D_MODEL + d];
  cpu_matmul(cross_feats.data(), Wk_x, Kc.data(), CROSS_M, D_MODEL, D_MODEL);
  cpu_matmul(cross_feats.data(), Wv_x, Vc.data(), CROSS_M, D_MODEL, D_MODEL);
  for(int i = 0; i < CROSS_M; i++)
    for(int d = 0; d < D_MODEL; d++)
      KTc[d*CROSS_M + i] = Kc[i*D_MODEL + d];

  cpu_matmul(Qc.data(), KTc.data(), sc.data(), 1, D_MODEL, CROSS_M);
  for(int i = 0; i < CROSS_M; i++) sc[i] *= sf;
  cpu_softmax(sc.data(), 1, CROSS_M);
  for(int d = 0; d < D_MODEL; d++) {
    float s = 0.f;
    for(int i = 0; i < CROSS_M; i++)
      s += sc[i] * Vc[i*D_MODEL + d];
    ctxt[d] = s;
  }
  std::vector<float> cross_out(D_MODEL);
  cpu_matmul(ctxt.data(), Wo_x, cross_out.data(), 1, D_MODEL, D_MODEL);
  for(int d = 0; d < D_MODEL; d++)
    X[(T-1)*D_MODEL + d] += cross_out[d];

  // Feed‑forward
  std::vector<float> f1(D_FF), f2(D_MODEL);
  for(int i = 0; i < D_FF; i++){
    float s = 0.f;
    for(int d = 0; d < D_MODEL; d++)
      s += X[(T-1)*D_MODEL + d] * W1[d*D_FF + i];
    f1[i] = s > 0 ? s : 0;
  }
  for(int d = 0; d < D_MODEL; d++){
    float s = 0.f;
    for(int i = 0; i < D_FF; i++)
      s += f1[i] * W2[i*D_MODEL + d];
    X[(T-1)*D_MODEL + d] += s;
  }

  // Project to vocab & argmax
  std::vector<float> logits(VOCAB_SIZE);
  for(int v = 0; v < VOCAB_SIZE; v++){
    float s = 0.f;
    for(int d = 0; d < D_MODEL; d++)
      s += X[(T-1)*D_MODEL + d] * Wout[d*VOCAB_SIZE + v];
    logits[v] = s;
  }
  return std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
}

// Decoder
void run_cpu_decoder(
  const std::vector<float>& cross_feats,
  const std::vector<float>& token_emb,
  const std::vector<float>& Wq_s,
  const std::vector<float>& Wk_s,
  const std::vector<float>& Wv_s,
  const std::vector<float>& Wo_s,
  const std::vector<float>& Wq_x,
  const std::vector<float>& Wk_x,
  const std::vector<float>& Wv_x,
  const std::vector<float>& Wo_x,
  const std::vector<float>& W1,
  const std::vector<float>& W2,
  const std::vector<float>& Wout
) {
  auto t0 = std::chrono::high_resolution_clock::now();
  std::vector<int> ids = {1}; // <START>=1, <END>=2
  while ((int)ids.size() < MAX_TEXT_LEN && ids.back() != 2) {
    ids.push_back(cpu_decoder_step(
      cross_feats, ids, token_emb.data(),
      Wq_s.data(), Wk_s.data(), Wv_s.data(), Wo_s.data(),
      Wq_x.data(), Wk_x.data(), Wv_x.data(), Wo_x.data(),
      W1.data(), W2.data(), Wout.data()
    ));
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double,std::milli>(t1 - t0).count();
  std::cout << "Total CPU inference time (ms): " << ms << " ms\n";
}

// GPU kernels

// Matrix multiplication: C[M×N] = A[M×K] × B[K×N]
__global__ void tiledMatMulKernel(
  const float *A,const float *B,float *C,
  int M,int K,int N
){
  __shared__ float sA[TILE_WIDTH][TILE_WIDTH],
                   sB[TILE_WIDTH][TILE_WIDTH];
  int row = blockIdx.y*TILE_WIDTH + threadIdx.y;
  int col = blockIdx.x*TILE_WIDTH + threadIdx.x;
  float val = 0.f;
  for(int t=0; t<(K+TILE_WIDTH-1)/TILE_WIDTH; t++){
    int aCol = t*TILE_WIDTH + threadIdx.x;
    int bRow = t*TILE_WIDTH + threadIdx.y;
    sA[threadIdx.y][threadIdx.x] =
      (row<M && aCol<K)? A[row*K + aCol] : 0.f;
    sB[threadIdx.y][threadIdx.x] =
      (bRow<K && col<N)? B[bRow*N + col] : 0.f;
    __syncthreads();
    for(int i=0;i<TILE_WIDTH;i++)
      val += sA[threadIdx.y][i] * sB[i][threadIdx.x];
    __syncthreads();
  }
  if(row<M && col<N) C[row*N + col] = val;
}

// Transpose matrix
__global__ void transposeKernel(
  const float *A,float *out,int rows,int cols
){
  int r = blockIdx.y*blockDim.y + threadIdx.y;
  int c = blockIdx.x*blockDim.x + threadIdx.x;
  if(r<rows && c<cols)
    out[c*rows + r] = A[r*cols + c];
}

// Scale matrix by a constant
__global__ void scaleKernel(float *m,int sz,float f){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<sz) m[i] *= f;
}

// Row-wise softmax
__global__ void softmaxKernel(float *mat,int rows,int cols){
  int r = blockIdx.x;
  extern __shared__ float buf[];
  float *bufMax = buf, *bufSum = buf+1;
  if(threadIdx.x==0){
    float m = mat[r*cols], s=0.f;
    for(int j=1;j<cols;j++) if(mat[r*cols+j]>m) m = mat[r*cols+j];
    *bufMax = m;
    for(int j=0;j<cols;j++){
      float e = expf(mat[r*cols+j] - m);
      mat[r*cols+j] = e;
      s += e;
    }
    *bufSum = s;
  }
  __syncthreads();
  float S = *bufSum;
  for(int j=threadIdx.x; j<cols; j+=blockDim.x)
    mat[r*cols+j] /= S;
}

// Causal mask for self-attention
__global__ void causalMaskKernel(float *s,int M){
  int i = blockIdx.y*blockDim.y + threadIdx.y;
  int j = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<M && j<M && j>i)
    s[i*M + j] = -1e9f;
}

// Add two vectors in place
__global__ void addInPlace(float *A,const float *B,int N){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<N) A[i] += B[i];
}

// ReLU activation
__global__ void reluKernel(float *A,int N){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<N && A[i]<0) A[i] = 0;
}

// Extracting query
__global__ void extractQuery(const float *seq,int t,float *Qd){
  int d = blockIdx.x*blockDim.x + threadIdx.x;
  if(d<D_MODEL)
    Qd[d] = seq[t*D_MODEL + d];
}

// Argmax
__global__ void argmaxKernel(const float *logits,int V,int *out){
  extern __shared__ float buf[];
  float *bestVal = buf;
  int   *bestIdx = (int*)(buf + blockDim.x);
  int tid = threadIdx.x;
  float bv = -1e20f; int bi = 0;
  for(int i=tid; i<V; i+=blockDim.x){
    float v = logits[i];
    if(v>bv){ bv = v; bi = i; }
  }
  bestVal[tid] = bv;
  bestIdx[tid] = bi;
  __syncthreads();
  for(int s=blockDim.x/2; s>0; s>>=1){
    if(tid<s && bestVal[tid+s] > bestVal[tid]){
      bestVal[tid] = bestVal[tid+s];
      bestIdx[tid] = bestIdx[tid+s];
    }
    __syncthreads();
  }
  if(tid==0) out[0] = bestIdx[0];
}

// Embed token
__global__ void embedToken(const int *ids, const float *tok, float *seq, int t){
  int d = blockIdx.x*blockDim.x + threadIdx.x;
  if(d<D_MODEL){
    int id = ids[t];
    seq[t*D_MODEL + d] = tok[id*D_MODEL + d];
  }
}

// Write id to output
__global__ void writeId(int *ids, const int *nid, int t){
  if(threadIdx.x==0) ids[t] = nid[0];
}

// Decoder
void run_gpu_decoder(
  const std::vector<float>& h_cross_feats,
  const std::vector<float>& h_token_emb,
  const std::vector<float>& h_Wq_s,
  const std::vector<float>& h_Wk_s,
  const std::vector<float>& h_Wv_s,
  const std::vector<float>& h_Wo_s,
  const std::vector<float>& h_Wq_x,
  const std::vector<float>& h_Wk_x,
  const std::vector<float>& h_Wv_x,
  const std::vector<float>& h_Wo_x,
  const std::vector<float>& h_W1,
  const std::vector<float>& h_W2,
  const std::vector<float>& h_Wout
) {
  // Allocate device buffers
  float *d_cross, *d_tok;
  CUDA_CHECK(cudaMalloc(&d_cross, CROSS_M*D_MODEL*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_tok,   VOCAB_SIZE*D_MODEL*sizeof(float)));

  int *d_ids, *d_next;
  float *d_seq;
  CUDA_CHECK(cudaMalloc(&d_ids,  MAX_TEXT_LEN*sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_next, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_seq,  MAX_TEXT_LEN*D_MODEL*sizeof(float)));

  // Weight buffers
  float *d_Wq_s, *d_Wk_s, *d_Wv_s, *d_Wo_s;
  float *d_Wq_x, *d_Wk_x, *d_Wv_x, *d_Wo_x;
  float *d_W1,   *d_W2,   *d_Wout;
  #define ALLOC(ptr,N) CUDA_CHECK(cudaMalloc(&ptr,(N)*sizeof(float)))
  ALLOC(d_Wq_s, D_MODEL*D_MODEL);
  ALLOC(d_Wk_s, D_MODEL*D_MODEL);
  ALLOC(d_Wv_s, D_MODEL*D_MODEL);
  ALLOC(d_Wo_s, D_MODEL*D_MODEL);
  ALLOC(d_Wq_x, D_MODEL*D_MODEL);
  ALLOC(d_Wk_x, D_MODEL*D_MODEL);
  ALLOC(d_Wv_x, D_MODEL*D_MODEL);
  ALLOC(d_Wo_x, D_MODEL*D_MODEL);
  ALLOC(d_W1,   D_MODEL*D_FF);
  ALLOC(d_W2,   D_FF*D_MODEL);
  ALLOC(d_Wout, D_MODEL*VOCAB_SIZE);

  // Intermediate buffers
  float *d_dec_Q, *d_dec_K, *d_dec_V, *d_dec_KT, *d_dec_scores, *d_dec_attn, *d_self_proj;
  float *d_Qd, *d_Kc, *d_Vc, *d_KcT, *d_sc, *d_ctxt, *d_cross_out;
  float *d_ffn1, *d_ffn2, *d_logits;
  ALLOC(d_dec_Q,      MAX_TEXT_LEN*D_MODEL);
  ALLOC(d_dec_K,      MAX_TEXT_LEN*D_MODEL);
  ALLOC(d_dec_V,      MAX_TEXT_LEN*D_MODEL);
  ALLOC(d_dec_KT,     D_MODEL*MAX_TEXT_LEN);
  ALLOC(d_dec_scores, MAX_TEXT_LEN*MAX_TEXT_LEN);
  ALLOC(d_dec_attn,   MAX_TEXT_LEN*D_MODEL);
  ALLOC(d_self_proj,  MAX_TEXT_LEN*D_MODEL);
  ALLOC(d_Qd,         D_MODEL);
  ALLOC(d_Kc,         CROSS_M*D_MODEL);
  ALLOC(d_Vc,         CROSS_M*D_MODEL);
  ALLOC(d_KcT,        D_MODEL*CROSS_M);
  ALLOC(d_sc,         CROSS_M);
  ALLOC(d_ctxt,       D_MODEL);
  ALLOC(d_cross_out,  D_MODEL);
  ALLOC(d_ffn1,       D_FF);
  ALLOC(d_ffn2,       D_MODEL);
  ALLOC(d_logits,     VOCAB_SIZE);

  // Timing events
  cudaEvent_t t0,t1,c0,c1,k0,k1,d0,d1;
  CUDA_CHECK(cudaEventCreate(&t0));
  CUDA_CHECK(cudaEventCreate(&t1));
  CUDA_CHECK(cudaEventCreate(&c0));
  CUDA_CHECK(cudaEventCreate(&c1));
  CUDA_CHECK(cudaEventCreate(&k0));
  CUDA_CHECK(cudaEventCreate(&k1));
  CUDA_CHECK(cudaEventCreate(&d0));
  CUDA_CHECK(cudaEventCreate(&d1));

  // Host to device copy
  CUDA_CHECK(cudaEventRecord(t0));
  CUDA_CHECK(cudaEventRecord(c0));
  CUDA_CHECK(cudaMemcpy(d_cross,  h_cross_feats.data(), CROSS_M*D_MODEL*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_tok,    h_token_emb.data(),   VOCAB_SIZE*D_MODEL*sizeof(float),   cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Wq_s,   h_Wq_s.data(),        D_MODEL*D_MODEL*sizeof(float),      cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Wk_s,   h_Wk_s.data(),        D_MODEL*D_MODEL*sizeof(float),      cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Wv_s,   h_Wv_s.data(),        D_MODEL*D_MODEL*sizeof(float),      cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Wo_s,   h_Wo_s.data(),        D_MODEL*D_MODEL*sizeof(float),      cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Wq_x,   h_Wq_x.data(),        D_MODEL*D_MODEL*sizeof(float),      cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Wk_x,   h_Wk_x.data(),        D_MODEL*D_MODEL*sizeof(float),      cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Wv_x,   h_Wv_x.data(),        D_MODEL*D_MODEL*sizeof(float),      cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Wo_x,   h_Wo_x.data(),        D_MODEL*D_MODEL*sizeof(float),      cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_W1,     h_W1.data(),          D_MODEL*D_FF*sizeof(float),         cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_W2,     h_W2.data(),          D_FF*D_MODEL*sizeof(float),         cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Wout,   h_Wout.data(),        D_MODEL*VOCAB_SIZE*sizeof(float),   cudaMemcpyHostToDevice));
  int start = 1;
  CUDA_CHECK(cudaMemcpy(d_ids, &start, sizeof(int), cudaMemcpyHostToDevice));
  {
    int blocks = (D_MODEL+255)/256;
    embedToken<<<blocks,256>>>(d_ids, d_tok, d_seq, 0);
  }
  CUDA_CHECK(cudaEventRecord(c1));

  // Kernels
  CUDA_CHECK(cudaEventRecord(k0));
  for (int t = 1; t < MAX_TEXT_LEN; t++) {
    dim3 bb(TILE_WIDTH, TILE_WIDTH);
  
    // Self‑Attention
    dim3 gbQ((D_MODEL + TILE_WIDTH - 1) / TILE_WIDTH, (t + TILE_WIDTH - 1) / TILE_WIDTH);
    tiledMatMulKernel<<<gbQ, bb>>>(d_seq,   d_Wq_s,    d_dec_Q,      t, D_MODEL, D_MODEL);
    tiledMatMulKernel<<<gbQ, bb>>>(d_seq,   d_Wk_s,    d_dec_K,      t, D_MODEL, D_MODEL);
    tiledMatMulKernel<<<gbQ, bb>>>(d_seq,   d_Wv_s,    d_dec_V,      t, D_MODEL, D_MODEL);
  
    dim3 tg((D_MODEL + TILE_WIDTH - 1) / TILE_WIDTH, (t + TILE_WIDTH - 1) / TILE_WIDTH);
    transposeKernel<<<tg, bb>>>(d_dec_K, d_dec_KT,    t, D_MODEL);
  
    dim3 gs((t + TILE_WIDTH - 1) / TILE_WIDTH, (t + TILE_WIDTH - 1) / TILE_WIDTH);
    tiledMatMulKernel<<<gs, bb>>>(d_dec_Q, d_dec_KT,   d_dec_scores, t, D_MODEL, t);
  
    causalMaskKernel<<<gs, bb>>>(d_dec_scores, t);
  
    scaleKernel<<<(t * t + 255) / 256, 256>>>(d_dec_scores, t * t, 1.0f / sqrtf((float)D_HEAD));
    softmaxKernel<<<t, 256, 2 * sizeof(float)>>>(d_dec_scores, t, t);
  
    tiledMatMulKernel<<<gs, bb>>>(d_dec_scores, d_dec_V, d_dec_attn, t, t, D_MODEL);
    tiledMatMulKernel<<<gbQ, bb>>>(d_dec_attn,  d_Wo_s,    d_self_proj, t, D_MODEL, D_MODEL);
    addInPlace<<<(t * D_MODEL + 255) / 256, 256>>>(d_seq, d_self_proj, t * D_MODEL);
  
    // Cross‑Attention
    extractQuery<<<(D_MODEL + 255) / 256, 256>>>(d_seq, t - 1, d_Qd);
  
    dim3 gbC((CROSS_M + TILE_WIDTH - 1) / TILE_WIDTH, (D_MODEL + TILE_WIDTH - 1) / TILE_WIDTH);
    tiledMatMulKernel<<<gbC, bb>>>(d_cross, d_Wk_x, d_Kc, CROSS_M, D_MODEL, D_MODEL);
    tiledMatMulKernel<<<gbC, bb>>>(d_cross, d_Wv_x, d_Vc, CROSS_M, D_MODEL, D_MODEL);
  
    dim3 tgC((D_MODEL + TILE_WIDTH - 1) / TILE_WIDTH, (CROSS_M + TILE_WIDTH - 1) / TILE_WIDTH);
    transposeKernel<<<tgC, bb>>>(d_Kc, d_KcT, CROSS_M, D_MODEL);
  
    dim3 gsC((CROSS_M + TILE_WIDTH - 1) / TILE_WIDTH, 1);
    tiledMatMulKernel<<<gsC, bb>>>(d_Qd, d_KcT, d_sc,    1, D_MODEL, CROSS_M);
    scaleKernel<<<(CROSS_M + 255) / 256, 256>>>(d_sc, CROSS_M, 1.0f / sqrtf((float)D_HEAD));
    softmaxKernel<<<1, 256, 2 * sizeof(float)>>>(d_sc, 1, CROSS_M);
  
    tiledMatMulKernel<<<(D_MODEL + TILE_WIDTH - 1) / TILE_WIDTH, bb>>>(d_sc, d_Vc, d_ctxt, 1, CROSS_M, D_MODEL);
    tiledMatMulKernel<<<(D_MODEL + TILE_WIDTH - 1) / TILE_WIDTH, bb>>>(d_ctxt, d_Wo_x, d_cross_out, 1, D_MODEL, D_MODEL);
    addInPlace<<<(D_MODEL + 255) / 256, 256>>>(d_seq + (t - 1) * D_MODEL, d_cross_out, D_MODEL);
  
    // Feed‑Forward
    tiledMatMulKernel<<<(D_FF + TILE_WIDTH - 1) / TILE_WIDTH, bb>>>(d_seq + (t - 1) * D_MODEL, d_W1,  d_ffn1, 1, D_MODEL, D_FF);
    reluKernel<<<(D_FF + 255) / 256, 256>>>(d_ffn1, D_FF);
    tiledMatMulKernel<<<(D_MODEL + TILE_WIDTH - 1) / TILE_WIDTH, bb>>>(d_ffn1, d_W2,  d_ffn2, 1, D_FF,    D_MODEL);
    addInPlace<<<(D_MODEL + 255) / 256, 256>>>(d_seq + (t - 1) * D_MODEL, d_ffn2, D_MODEL);
  
    // Logits & Argmax
    tiledMatMulKernel<<<(VOCAB_SIZE + TILE_WIDTH - 1) / TILE_WIDTH, bb>>>(d_seq + (t - 1) * D_MODEL, d_Wout, d_logits, 1, D_MODEL, VOCAB_SIZE);
    argmaxKernel<<<1, 1024, 1024 * (sizeof(float) + sizeof(int))>>>(d_logits, VOCAB_SIZE, d_next);
    writeId<<<1, 1>>>(d_ids, d_next, t);
  
    // Embed Next Token
    embedToken<<<(D_MODEL + 255) / 256, 256>>>(d_ids, d_tok, d_seq, t);
  }
  
  CUDA_CHECK(cudaEventRecord(k1));

  // Device to host copy
  CUDA_CHECK(cudaEventRecord(d0));
  std::vector<int> h_ids(MAX_TEXT_LEN);
  CUDA_CHECK(cudaMemcpy(h_ids.data(), d_ids, MAX_TEXT_LEN*sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(d1));
  CUDA_CHECK(cudaEventSynchronize(d1));

  // End timing
  CUDA_CHECK(cudaEventRecord(t1));
  CUDA_CHECK(cudaEventSynchronize(t1));

  float t_h2d, t_kern, t_d2h, t_tot;
  CUDA_CHECK(cudaEventElapsedTime(&t_h2d, c0, c1));
  CUDA_CHECK(cudaEventElapsedTime(&t_kern, k0, k1));
  CUDA_CHECK(cudaEventElapsedTime(&t_d2h, d0, d1));
  CUDA_CHECK(cudaEventElapsedTime(&t_tot,  t0, t1));

  std::cout << "GPU timings (ms):\n"
            << "  Host-to-Device copy time: " << t_h2d  << "\n"
            << "  Kernel execution time:    " << t_kern << "\n"
            << "  Device-to-Host copy time: " << t_d2h  << "\n"
            << "  Total GPU time:           " << t_tot  << "\n";

  // Cleanup
  cudaFree(d_cross); cudaFree(d_tok);
  cudaFree(d_ids);   cudaFree(d_next); cudaFree(d_seq);
  cudaFree(d_Wq_s); cudaFree(d_Wk_s);
  cudaFree(d_Wv_s); cudaFree(d_Wo_s);
  cudaFree(d_Wq_x); cudaFree(d_Wk_x);
  cudaFree(d_Wv_x); cudaFree(d_Wo_x);
  cudaFree(d_W1);   cudaFree(d_W2);
  cudaFree(d_Wout);

  cudaFree(d_dec_Q);      cudaFree(d_dec_K);
  cudaFree(d_dec_V);      cudaFree(d_dec_KT);
  cudaFree(d_dec_scores); cudaFree(d_dec_attn);
  cudaFree(d_self_proj);

  cudaFree(d_Qd);  cudaFree(d_Kc);
  cudaFree(d_Vc);  cudaFree(d_KcT);
  cudaFree(d_sc);  cudaFree(d_ctxt);
  cudaFree(d_cross_out);

  cudaFree(d_ffn1); cudaFree(d_ffn2);
  cudaFree(d_logits);

  cudaEventDestroy(t0); cudaEventDestroy(t1);
  cudaEventDestroy(c0); cudaEventDestroy(c1);
  cudaEventDestroy(k0); cudaEventDestroy(k1);
  cudaEventDestroy(d0); cudaEventDestroy(d1);
}

int main(){
  // Prepare random data
  std::vector<float> cross_feats(CROSS_M*D_MODEL),
                     token_emb(VOCAB_SIZE*D_MODEL),
                     Wq_s(D_MODEL*D_MODEL), Wk_s(D_MODEL*D_MODEL),
                     Wv_s(D_MODEL*D_MODEL), Wo_s(D_MODEL*D_MODEL),
                     Wq_x(D_MODEL*D_MODEL), Wk_x(D_MODEL*D_MODEL),
                     Wv_x(D_MODEL*D_MODEL), Wo_x(D_MODEL*D_MODEL),
                     W1(D_MODEL*D_FF), W2(D_FF*D_MODEL),
                     Wout(D_MODEL*VOCAB_SIZE);
  srand(42);
  for(auto &x:cross_feats) x = rand()/float(RAND_MAX);
  for(auto &x:token_emb)  x = rand()/float(RAND_MAX);
  for(auto &x:Wq_s) x = rand()/float(RAND_MAX);
  for(auto &x:Wk_s) x = rand()/float(RAND_MAX);
  for(auto &x:Wv_s) x = rand()/float(RAND_MAX);
  for(auto &x:Wo_s) x = rand()/float(RAND_MAX);
  for(auto &x:Wq_x) x = rand()/float(RAND_MAX);
  for(auto &x:Wk_x) x = rand()/float(RAND_MAX);
  for(auto &x:Wv_x) x = rand()/float(RAND_MAX);
  for(auto &x:Wo_x) x = rand()/float(RAND_MAX);
  for(auto &x:W1)   x = rand()/float(RAND_MAX);
  for(auto &x:W2)   x = rand()/float(RAND_MAX);
  for(auto &x:Wout) x = rand()/float(RAND_MAX);

  run_cpu_decoder(cross_feats, token_emb,
                  Wq_s,Wk_s,Wv_s,Wo_s,
                  Wq_x,Wk_x,Wv_x,Wo_x,
                  W1,W2,Wout);

  run_gpu_decoder(cross_feats, token_emb,
                  Wq_s,Wk_s,Wv_s,Wo_s,
                  Wq_x,Wk_x,Wv_x,Wo_x,
                  W1,W2,Wout);

  return 0;
}
