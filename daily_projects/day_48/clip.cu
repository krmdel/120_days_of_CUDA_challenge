 #include <cuda_runtime.h>
 #include <iostream>
 #include <chrono>
 #include <cmath>
 #include <cstdlib>
 #include <algorithm>
 
 #define CUDA_CHECK(x)                                                        \
   do {                                                                       \
     cudaError_t err = (x);                                                   \
     if (err != cudaSuccess) {                                                \
       std::cerr << "CUDA error: " << cudaGetErrorString(err)                 \
                 << " at " << __FILE__ << ":" << __LINE__ << std::endl;       \
       std::exit(EXIT_FAILURE);                                               \
     }                                                                        \
   } while (0)
 
 // Image parameters
 #define IMG_H            1024
 #define IMG_W            1024
 #define PATCH            32
 #define NUM_PH           (IMG_H / PATCH)
 #define NUM_PW           (IMG_W / PATCH)
 #define NUM_PATCHES      (NUM_PH * NUM_PW)        /* 1024                   */
 #define PATCH_VEC        (PATCH * PATCH)          /* 32×32                  */
 
 // Text parameters
 #define SEQ_LEN          256
 
 /* Model width */
 #define D_MODEL          256
 #define H_HEADS          8
 #define D_HEAD           (D_MODEL / H_HEADS)
 #define D_FF             (4 * D_MODEL)
 
 // Tiling
 #define TILE_WIDTH       16
 
 
 // CPU functions
 
 // Matrix multiplication
 void cpu_matmul(const float* A,const float* B,float* C,int M,int K,int N)
 {
   for (int i = 0; i < M; ++i)
     for (int j = 0; j < N; ++j) {
       float s = 0.f;
       for (int k = 0; k < K; ++k)
         s += A[i*K+k] * B[k*N+j];
       C[i*N+j] = s;
     }
 }
 
 // Softmax
 void cpu_softmax(float* M,int rows,int cols)
 {
   for (int r = 0; r < rows; ++r) {
     float* row = M + r*cols;
     float m = row[0];
     for (int j = 1; j < cols; ++j) if (row[j] > m) m = row[j];
     float s = 0.f;
     for (int j = 0; j < cols; ++j) { row[j] = std::exp(row[j]-m); s += row[j]; }
     for (int j = 0; j < cols; ++j)   row[j] /= s;
   }
 }
 
 // Patch embedding
 void cpu_patch_embed(const float* img,const float* W,float* out)
 {
   for (int p = 0; p < NUM_PATCHES; ++p) {
     int pr = p / NUM_PW, pc = p % NUM_PW;
     int sr = pr * PATCH,  sc = pc * PATCH;
     for (int d = 0; d < D_MODEL; ++d) {
       float acc = 0.f;
       for (int i = 0; i < PATCH; ++i)
         for (int j = 0; j < PATCH; ++j) {
           float pix = img[(sr+i)*IMG_W + sc+j];
           acc += pix * W[((i*PATCH)+j)*D_MODEL + d];
         }
       out[p*D_MODEL+d] = acc;
     }
   }
   for (int p = 0; p < NUM_PATCHES; ++p)
     for (int d = 0; d < D_MODEL; ++d) {
       float ang = p / std::pow(10000.f,(2*(d/2))/(float)D_MODEL);
       out[p*D_MODEL+d] += (d&1) ? std::cos(ang) : std::sin(ang);
     }
 }
 
 // Image encoder
 void cpu_img_encoder(const float* X,const float* Wq,const float* Wk,
                      const float* Wv,const float* Wo,float* out)
 {
   float *Q = new float[NUM_PATCHES*D_MODEL];
   float *K = new float[NUM_PATCHES*D_MODEL];
   float *V = new float[NUM_PATCHES*D_MODEL];
   float *KT= new float[NUM_PATCHES*D_MODEL];
   float *S = new float[NUM_PATCHES*NUM_PATCHES];
   float *A = new float[NUM_PATCHES*D_MODEL];
 
   cpu_matmul(X,Wq,Q,NUM_PATCHES,D_MODEL,D_MODEL);
   cpu_matmul(X,Wk,K,NUM_PATCHES,D_MODEL,D_MODEL);
   cpu_matmul(X,Wv,V,NUM_PATCHES,D_MODEL,D_MODEL);
 
   for(int i=0;i<NUM_PATCHES;i++)
     for(int d=0;d<D_MODEL;d++)
       KT[d*NUM_PATCHES+i]=K[i*D_MODEL+d];
 
   cpu_matmul(Q,KT,S,NUM_PATCHES,D_MODEL,NUM_PATCHES);
   float sf=1.f/std::sqrt((float)D_MODEL);
   for(int i=0;i<NUM_PATCHES*NUM_PATCHES;i++) S[i]*=sf;
   cpu_softmax(S,NUM_PATCHES,NUM_PATCHES);
   cpu_matmul(S,V,A,NUM_PATCHES,NUM_PATCHES,D_MODEL);
   cpu_matmul(A,Wo,out,NUM_PATCHES,D_MODEL,D_MODEL);
 
   delete[] Q; delete[] K; delete[] V; delete[] KT; delete[] S; delete[] A;
 }
 
 // Ppooling & normalization
 void cpu_pool_norm(const float* seq,int tokens,float* dst)
 {
   for(int d=0;d<D_MODEL;d++){
     float s=0.f;
     for(int t=0;t<tokens;t++) s+=seq[t*D_MODEL+d];
     dst[d]=s/tokens;
   }
   float ss=0.f;
   for(int d=0;d<D_MODEL;d++) ss+=dst[d]*dst[d];
   float inv=1.f/std::sqrt(ss+1e-8f);
   for(int d=0;d<D_MODEL;d++) dst[d]*=inv;
 }
 
 // Text encoder
 void cpu_text_encoder(const float* X,const float* Wq,const float* Wk,
                       const float* Wv,const float* Wo,const float* W1,
                       const float* W2,float* out)
 {
   float *Q=new float[SEQ_LEN*D_MODEL],
         *K=new float[SEQ_LEN*D_MODEL],
         *V=new float[SEQ_LEN*D_MODEL],
         *KT=new float[SEQ_LEN*D_MODEL],
         *S=new float[SEQ_LEN*SEQ_LEN],
         *A=new float[SEQ_LEN*D_MODEL],
         *Proj=new float[SEQ_LEN*D_MODEL],
         *Add=new float[SEQ_LEN*D_MODEL],
         *Norm=new float[SEQ_LEN*D_MODEL],
         *FF1=new float[SEQ_LEN*D_FF],
         *FF2=new float[SEQ_LEN*D_MODEL];
 
   cpu_matmul(X,Wq,Q,SEQ_LEN,D_MODEL,D_MODEL);
   cpu_matmul(X,Wk,K,SEQ_LEN,D_MODEL,D_MODEL);
   cpu_matmul(X,Wv,V,SEQ_LEN,D_MODEL,D_MODEL);
   for(int i=0;i<SEQ_LEN;i++)
     for(int d=0;d<D_MODEL;d++)
       KT[d*SEQ_LEN+i]=K[i*D_MODEL+d];
 
   cpu_matmul(Q,KT,S,SEQ_LEN,D_MODEL,SEQ_LEN);
   float sf=1.f/std::sqrt((float)D_HEAD);
   for(int i=0;i<SEQ_LEN*SEQ_LEN;i++) S[i]*=sf;
   cpu_softmax(S,SEQ_LEN,SEQ_LEN);
   cpu_matmul(S,V,A,SEQ_LEN,SEQ_LEN,D_MODEL);
   cpu_matmul(A,Wo,Proj,SEQ_LEN,D_MODEL,D_MODEL);
 
   /* residual + LN */
   for(int i=0;i<SEQ_LEN*D_MODEL;i++) Add[i]=X[i]+Proj[i];
   for(int t=0;t<SEQ_LEN;t++){
     float mean=0.f,var=0.f;
     for(int d=0;d<D_MODEL;d++) mean+=Add[t*D_MODEL+d];
     mean/=D_MODEL;
     for(int d=0;d<D_MODEL;d++){ float diff=Add[t*D_MODEL+d]-mean; var+=diff*diff;}
     var/=D_MODEL; float inv=1.f/std::sqrt(var+1e-5f);
     for(int d=0;d<D_MODEL;d++) Norm[t*D_MODEL+d]=(Add[t*D_MODEL+d]-mean)*inv;
   }
 
   cpu_matmul(Norm,W1,FF1,SEQ_LEN,D_MODEL,D_FF);
   for(int i=0;i<SEQ_LEN*D_FF;i++) if(FF1[i]<0) FF1[i]=0;
   cpu_matmul(FF1,W2,FF2,SEQ_LEN,D_FF,D_MODEL);
 
   for(int i=0;i<SEQ_LEN*D_MODEL;i++) Add[i]=Norm[i]+FF2[i];
   for(int t=0;t<SEQ_LEN;t++){
     float mean=0.f,var=0.f;
     for(int d=0;d<D_MODEL;d++) mean+=Add[t*D_MODEL+d];
     mean/=D_MODEL;
     for(int d=0;d<D_MODEL;d++){ float diff=Add[t*D_MODEL+d]-mean; var+=diff*diff;}
     var/=D_MODEL; float inv=1.f/std::sqrt(var+1e-5f);
     for(int d=0;d<D_MODEL;d++) out[t*D_MODEL+d]=(Add[t*D_MODEL+d]-mean)*inv;
   }
 
   delete[] Q; delete[] K; delete[] V; delete[] KT; delete[] S;
   delete[] A; delete[] Proj; delete[] Add; delete[] Norm;
   delete[] FF1; delete[] FF2;
 }
 
// CUDA kernels

// Tiled matrix multiplication
 __global__
 void tiledMatMulKernel(const float* A,const float* B,float* C,
                        int M,int K,int N)
 {
   __shared__ float sA[TILE_WIDTH][TILE_WIDTH];
   __shared__ float sB[TILE_WIDTH][TILE_WIDTH];
   int row=blockIdx.y*TILE_WIDTH+threadIdx.y;
   int col=blockIdx.x*TILE_WIDTH+threadIdx.x;
   float val=0.f;
   for(int t=0;t<(K+TILE_WIDTH-1)/TILE_WIDTH;t++){
     int aCol=t*TILE_WIDTH+threadIdx.x;
     int bRow=t*TILE_WIDTH+threadIdx.y;
     sA[threadIdx.y][threadIdx.x]=(row<M&&aCol<K)?A[row*K+aCol]:0.f;
     sB[threadIdx.y][threadIdx.x]=(bRow<K&&col<N)?B[bRow*N+col]:0.f;
     __syncthreads();
     #pragma unroll
     for(int i=0;i<TILE_WIDTH;i++) val+=sA[threadIdx.y][i]*sB[i][threadIdx.x];
     __syncthreads();
   }
   if(row<M&&col<N) C[row*N+col]=val;
 }
 
 // Transpose
 __global__
 void transposeKernel(const float* in,float* out,int rows,int cols)
 {
   int r=blockIdx.y*blockDim.y+threadIdx.y;
   int c=blockIdx.x*blockDim.x+threadIdx.x;
   if(r<rows&&c<cols) out[c*rows+r]=in[r*cols+c];
 }
 
 // Scaling
 __global__
 void scaleKernel(float* M,int n,float factor)
 {
   int idx=blockIdx.x*blockDim.x+threadIdx.x;
   if(idx<n) M[idx]*=factor;
 }
 
 // Row‑wise softmax
 __global__
 void softmaxKernel(float* M,int rows,int cols)
 {
   int r=blockIdx.x;
   extern __shared__ float buf[];
   float* rowMax=buf; float* rowSum=buf+1;
   if(threadIdx.x==0){
     float m=M[r*cols], s=0.f;
     for(int j=1;j<cols;j++) if(M[r*cols+j]>m) m=M[r*cols+j];
     *rowMax=m;
     for(int j=0;j<cols;j++){ float e=expf(M[r*cols+j]-m); M[r*cols+j]=e; s+=e; }
     *rowSum=s;
   }
   __syncthreads();
   float S=*rowSum;
   for(int j=threadIdx.x;j<cols;j+=blockDim.x) M[r*cols+j]/=S;
 }
 
 // Patch‑embedding
 __global__
 void patch_embed_kernel(const float* img,const float* W,float* out)
 {
   int p=blockIdx.x;
   int tid=threadIdx.x;
   if(p>=NUM_PATCHES) return;
 
   __shared__ float sPatch[PATCH_VEC];
   if(tid<PATCH_VEC){
     int pr=p/NUM_PW, pc=p%NUM_PW;
     int sr=pr*PATCH, sc=pc*PATCH;
     int i=tid/PATCH, j=tid%PATCH;
     sPatch[tid] = img[(sr+i)*IMG_W + sc+j];
   }
   __syncthreads();
   if(tid<D_MODEL){
     float s=0.f;
     #pragma unroll
     for(int k=0;k<PATCH_VEC;k++)
       s += sPatch[k] * W[k*D_MODEL + tid];
     out[p*D_MODEL + tid] = s;
   }
 }
 
 // Positional embedding
 __global__
 void add_pos_kernel(float* emb)
 {
   int p=blockIdx.x, d=threadIdx.x;
   if(p<NUM_PATCHES && d<D_MODEL){
     float ang = p / powf(10000.f,(2*(d/2))/(float)D_MODEL);
     float pos = (d&1) ? cosf(ang) : sinf(ang);
     emb[p*D_MODEL+d] += pos;
   }
 }
 
 // Residual addition
 __global__
 void addInPlace(float* A,const float* B,int N)
 {
   int idx=blockIdx.x*blockDim.x+threadIdx.x;
   if(idx<N) A[idx]+=B[idx];
 }
 
 // ReLU activation
 __global__
 void reluKernel(float* A,int N)
 {
   int idx=blockIdx.x*blockDim.x+threadIdx.x;
   if(idx<N && A[idx]<0) A[idx]=0;
 }
 
 // Layer‑norm per row
 __global__
 void layer_norm_kernel(const float* input,float* output,int M,int N,float eps)
 {
   int row=blockIdx.x;
   extern __shared__ float sh[];
   int tid=threadIdx.x;
 
   float sum=0.f;
   for(int j=tid;j<N;j+=blockDim.x) sum+=input[row*N+j];
   sh[tid]=sum; __syncthreads();
   for(int s=blockDim.x/2;s>0;s>>=1){ if(tid<s) sh[tid]+=sh[tid+s]; __syncthreads();}
   float mean=sh[0]/N; __syncthreads();
 
   float varSum=0.f;
   for(int j=tid;j<N;j+=blockDim.x){
     float d=input[row*N+j]-mean;
     varSum+=d*d;
   }
   sh[tid]=varSum; __syncthreads();
   for(int s=blockDim.x/2;s>0;s>>=1){ if(tid<s) sh[tid]+=sh[tid+s]; __syncthreads();}
   float var=sh[0]/N;
   float inv=rsqrtf(var+eps);
 
   for(int j=tid;j<N;j+=blockDim.x)
     output[row*N+j]=(input[row*N+j]-mean)*inv;
 }
 
 // Mean‑pool
 __global__
 void mean_pool_kernel(const float* seq,float* vec)
 {
   int d=threadIdx.x;
   float s=0.f;
   for(int p=0;p<NUM_PATCHES;p++) s+=seq[p*D_MODEL+d];
   vec[d]=s/NUM_PATCHES;
 }
 
 // L2 norm
 __global__
 void l2_norm_kernel(float* vec)
 {
   __shared__ float ssq;
   if(threadIdx.x==0) ssq=0.f;
   __syncthreads();
   float v=vec[threadIdx.x];
   atomicAdd(&ssq,v*v);
   __syncthreads();
   float inv=rsqrtf(ssq+1e-8f);
   vec[threadIdx.x]*=inv;
 }
 
 // Dot product
 __global__
 void dot_kernel(const float* a,const float* b,float* out)
 {
   __shared__ float sm[256];
   int tid=threadIdx.x;
   sm[tid]=a[tid]*b[tid];
   __syncthreads();
   for(int s=blockDim.x/2;s>0;s>>=1){
     if(tid<s) sm[tid]+=sm[tid+s];
     __syncthreads();
   }
   if(tid==0) *out=sm[0];
 }
 
 int main()
 {
   std::srand(42);
 
   // Host allocations & random initilization
   size_t imgSize   = IMG_H*IMG_W;
   size_t peWeights = PATCH_VEC*D_MODEL;
   size_t encW      = D_MODEL*D_MODEL;
   size_t ffW1      = D_MODEL*D_FF;
   size_t ffW2      = D_FF*D_MODEL;
   size_t tokEmb    = 30522 * D_MODEL;
 
   float *h_img  = new float[imgSize];
   float *h_Wpe  = new float[peWeights];
   float *h_Wq_i = new float[encW];
   float *h_Wk_i = new float[encW];
   float *h_Wv_i = new float[encW];
   float *h_Wo_i = new float[encW];
 
   float *h_tok  = new float[tokEmb];
   float *h_Wq_t = new float[encW];
   float *h_Wk_t = new float[encW];
   float *h_Wv_t = new float[encW];
   float *h_Wo_t = new float[encW];
   float *h_W1   = new float[ffW1];
   float *h_W2   = new float[ffW2];
 
   for(size_t i=0;i<imgSize;i++)      h_img[i]  = (float)rand()/RAND_MAX;
   for(size_t i=0;i<peWeights;i++)    h_Wpe[i]  = (float)rand()/RAND_MAX;
   for(size_t i=0;i<encW;i++){
     h_Wq_i[i]=(float)rand()/RAND_MAX; h_Wk_i[i]=(float)rand()/RAND_MAX;
     h_Wv_i[i]=(float)rand()/RAND_MAX; h_Wo_i[i]=(float)rand()/RAND_MAX;
     h_Wq_t[i]=(float)rand()/RAND_MAX; h_Wk_t[i]=(float)rand()/RAND_MAX;
     h_Wv_t[i]=(float)rand()/RAND_MAX; h_Wo_t[i]=(float)rand()/RAND_MAX;
   }
   for(size_t i=0;i<ffW1;i++) h_W1[i]=(float)rand()/RAND_MAX;
   for(size_t i=0;i<ffW2;i++) h_W2[i]=(float)rand()/RAND_MAX;
   for(size_t i=0;i<tokEmb;i++) h_tok[i]=(float)rand()/RAND_MAX;
 
   // CPU inference
   auto c0=std::chrono::high_resolution_clock::now();
 
   float *patch_cpu=new float[NUM_PATCHES*D_MODEL];
   cpu_patch_embed(h_img,h_Wpe,patch_cpu);
   float *enc_cpu=new float[NUM_PATCHES*D_MODEL];
   cpu_img_encoder(patch_cpu,h_Wq_i,h_Wk_i,h_Wv_i,h_Wo_i,enc_cpu);
   float img_vec[D_MODEL]; cpu_pool_norm(enc_cpu,NUM_PATCHES,img_vec);
 
   float *seq_cpu=new float[SEQ_LEN*D_MODEL];
   for(int t=0;t<SEQ_LEN;t++)
     for(int d=0;d<D_MODEL;d++)
       seq_cpu[t*D_MODEL+d]=h_tok[(t%30522)*D_MODEL+d];
 
   float *txt_enc=new float[SEQ_LEN*D_MODEL];
   cpu_text_encoder(seq_cpu,h_Wq_t,h_Wk_t,h_Wv_t,h_Wo_t,h_W1,h_W2,txt_enc);
   float txt_vec[D_MODEL];
   for(int d=0;d<D_MODEL;d++) txt_vec[d]=txt_enc[d];
   cpu_pool_norm(txt_vec,1,txt_vec);
 
   float sim_cpu=0.f;
   for(int d=0;d<D_MODEL;d++) sim_cpu += img_vec[d]*txt_vec[d];
 
   auto c1=std::chrono::high_resolution_clock::now();
   double cpu_ms=std::chrono::duration<double,std::milli>(c1-c0).count();
 
   std::cout<<"CPU cosine similarity: "<<sim_cpu<<"\n";
   std::cout<<"Total CPU inference time (ms): "<<cpu_ms<<"\n";
 
   // GPU allocations
   float *d_img,*d_Wpe,*d_patch,*d_Wq_i,*d_Wk_i,*d_Wv_i,*d_Wo_i;
   float *d_tok,*d_Wq_t,*d_Wk_t,*d_Wv_t,*d_Wo_t,*d_W1,*d_W2;
   float *d_enc_Q,*d_enc_K,*d_enc_V,*d_enc_KT,*d_enc_S,*d_enc_A,*d_enc_O;
   float *d_emb_in,*d_txt_Q,*d_txt_K,*d_txt_V,*d_txt_KT,*d_txt_S,*d_txt_A,
         *d_txt_P,*d_norm1,*d_ff1,*d_ff2,*d_norm2;
   float *d_img_vec,*d_txt_vec,*d_sim;
 
   #define ALOC(ptr,n) CUDA_CHECK(cudaMalloc(&ptr,(n)*sizeof(float)))
   ALOC(d_img,imgSize);  ALOC(d_Wpe,peWeights); ALOC(d_patch,NUM_PATCHES*D_MODEL);
   ALOC(d_Wq_i,encW);    ALOC(d_Wk_i,encW);     ALOC(d_Wv_i,encW);  ALOC(d_Wo_i,encW);
 
   ALOC(d_tok,tokEmb);   ALOC(d_Wq_t,encW);     ALOC(d_Wk_t,encW);
   ALOC(d_Wv_t,encW);    ALOC(d_Wo_t,encW);     ALOC(d_W1,ffW1);    ALOC(d_W2,ffW2);
 
   ALOC(d_enc_Q,NUM_PATCHES*D_MODEL); ALOC(d_enc_K,NUM_PATCHES*D_MODEL);
   ALOC(d_enc_V,NUM_PATCHES*D_MODEL); ALOC(d_enc_KT,NUM_PATCHES*D_MODEL);
   ALOC(d_enc_S,NUM_PATCHES*NUM_PATCHES); ALOC(d_enc_A,NUM_PATCHES*D_MODEL);
   ALOC(d_enc_O,NUM_PATCHES*D_MODEL);
 
   ALOC(d_emb_in,SEQ_LEN*D_MODEL);    ALOC(d_txt_Q,SEQ_LEN*D_MODEL);
   ALOC(d_txt_K,SEQ_LEN*D_MODEL);     ALOC(d_txt_V,SEQ_LEN*D_MODEL);
   ALOC(d_txt_KT,SEQ_LEN*D_MODEL);    ALOC(d_txt_S,SEQ_LEN*SEQ_LEN);
   ALOC(d_txt_A,SEQ_LEN*D_MODEL);     ALOC(d_txt_P,SEQ_LEN*D_MODEL);
   ALOC(d_norm1,SEQ_LEN*D_MODEL);     ALOC(d_ff1,SEQ_LEN*D_FF);
   ALOC(d_ff2,SEQ_LEN*D_MODEL);       ALOC(d_norm2,SEQ_LEN*D_MODEL);
 
   CUDA_CHECK(cudaMalloc(&d_img_vec,D_MODEL*sizeof(float)));
   CUDA_CHECK(cudaMalloc(&d_txt_vec,D_MODEL*sizeof(float)));
   CUDA_CHECK(cudaMalloc(&d_sim,sizeof(float)));
 
   // Timing events
   cudaEvent_t t0,t1,h0,h1,k0,k1,d0,d1;
   CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
   CUDA_CHECK(cudaEventCreate(&h0)); CUDA_CHECK(cudaEventCreate(&h1));
   CUDA_CHECK(cudaEventCreate(&k0)); CUDA_CHECK(cudaEventCreate(&k1));
   CUDA_CHECK(cudaEventCreate(&d0)); CUDA_CHECK(cudaEventCreate(&d1));
 
   CUDA_CHECK(cudaEventRecord(t0));
   CUDA_CHECK(cudaEventRecord(h0));

   // Host-to-device copy
   CUDA_CHECK(cudaMemcpy(d_img ,h_img ,imgSize*sizeof(float),cudaMemcpyHostToDevice));
   CUDA_CHECK(cudaMemcpy(d_Wpe ,h_Wpe ,peWeights*sizeof(float),cudaMemcpyHostToDevice));
 
   CUDA_CHECK(cudaMemcpy(d_Wq_i,h_Wq_i,encW*sizeof(float),cudaMemcpyHostToDevice));
   CUDA_CHECK(cudaMemcpy(d_Wk_i,h_Wk_i,encW*sizeof(float),cudaMemcpyHostToDevice));
   CUDA_CHECK(cudaMemcpy(d_Wv_i,h_Wv_i,encW*sizeof(float),cudaMemcpyHostToDevice));
   CUDA_CHECK(cudaMemcpy(d_Wo_i,h_Wo_i,encW*sizeof(float),cudaMemcpyHostToDevice));
 
   CUDA_CHECK(cudaMemcpy(d_tok ,h_tok ,tokEmb*sizeof(float),cudaMemcpyHostToDevice));
   CUDA_CHECK(cudaMemcpy(d_Wq_t,h_Wq_t,encW*sizeof(float),cudaMemcpyHostToDevice));
   CUDA_CHECK(cudaMemcpy(d_Wk_t,h_Wk_t,encW*sizeof(float),cudaMemcpyHostToDevice));
   CUDA_CHECK(cudaMemcpy(d_Wv_t,h_Wv_t,encW*sizeof(float),cudaMemcpyHostToDevice));
   CUDA_CHECK(cudaMemcpy(d_Wo_t,h_Wo_t,encW*sizeof(float),cudaMemcpyHostToDevice));
 
   CUDA_CHECK(cudaMemcpy(d_W1,h_W1,ffW1*sizeof(float),cudaMemcpyHostToDevice));
   CUDA_CHECK(cudaMemcpy(d_W2,h_W2,ffW2*sizeof(float),cudaMemcpyHostToDevice));
 
   CUDA_CHECK(cudaMemcpy(d_emb_in,seq_cpu,SEQ_LEN*D_MODEL*sizeof(float),cudaMemcpyHostToDevice));
   CUDA_CHECK(cudaEventRecord(h1));
 
   // Kernel execution
   CUDA_CHECK(cudaEventRecord(k0));
 
   // Image
   patch_embed_kernel<<<NUM_PATCHES,D_MODEL>>>(d_img,d_Wpe,d_patch);
   add_pos_kernel<<<NUM_PATCHES,D_MODEL>>>(d_patch);
 
   dim3 bb(TILE_WIDTH,TILE_WIDTH);
   dim3 gb_q((D_MODEL+TILE_WIDTH-1)/TILE_WIDTH,(NUM_PATCHES+TILE_WIDTH-1)/TILE_WIDTH);
   tiledMatMulKernel<<<gb_q,bb>>>(d_patch,d_Wq_i,d_enc_Q,NUM_PATCHES,D_MODEL,D_MODEL);
   tiledMatMulKernel<<<gb_q,bb>>>(d_patch,d_Wk_i,d_enc_K,NUM_PATCHES,D_MODEL,D_MODEL);
   tiledMatMulKernel<<<gb_q,bb>>>(d_patch,d_Wv_i,d_enc_V,NUM_PATCHES,D_MODEL,D_MODEL);
 
   dim3 gbt((D_MODEL+TILE_WIDTH-1)/TILE_WIDTH,(NUM_PATCHES+TILE_WIDTH-1)/TILE_WIDTH);
   transposeKernel<<<gbt,bb>>>(d_enc_K,d_enc_KT,NUM_PATCHES,D_MODEL);
 
   dim3 gb_s((NUM_PATCHES+TILE_WIDTH-1)/TILE_WIDTH,(NUM_PATCHES+TILE_WIDTH-1)/TILE_WIDTH);
   tiledMatMulKernel<<<gb_s,bb>>>(d_enc_Q,d_enc_KT,d_enc_S,NUM_PATCHES,D_MODEL,NUM_PATCHES);
   scaleKernel<<<(NUM_PATCHES*NUM_PATCHES+255)/256,256>>>(d_enc_S,NUM_PATCHES*NUM_PATCHES,
                                                          1.f/std::sqrt((float)D_MODEL));
   softmaxKernel<<<NUM_PATCHES,256,2*sizeof(float)>>>(d_enc_S,NUM_PATCHES,NUM_PATCHES);
   tiledMatMulKernel<<<gb_s,bb>>>(d_enc_S,d_enc_V,d_enc_A,NUM_PATCHES,NUM_PATCHES,D_MODEL);
   tiledMatMulKernel<<<gb_q,bb>>>(d_enc_A,d_Wo_i,d_enc_O,NUM_PATCHES,D_MODEL,D_MODEL);
 
   mean_pool_kernel<<<1,D_MODEL>>>(d_enc_O,d_img_vec);
   l2_norm_kernel<<<1,D_MODEL>>>(d_img_vec);
 
   // Text
   dim3 gb_tx((D_MODEL+TILE_WIDTH-1)/TILE_WIDTH,(SEQ_LEN+TILE_WIDTH-1)/TILE_WIDTH);
   tiledMatMulKernel<<<gb_tx,bb>>>(d_emb_in,d_Wq_t,d_txt_Q,SEQ_LEN,D_MODEL,D_MODEL);
   tiledMatMulKernel<<<gb_tx,bb>>>(d_emb_in,d_Wk_t,d_txt_K,SEQ_LEN,D_MODEL,D_MODEL);
   tiledMatMulKernel<<<gb_tx,bb>>>(d_emb_in,d_Wv_t,d_txt_V,SEQ_LEN,D_MODEL,D_MODEL);
 
   dim3 gbt2((D_MODEL+TILE_WIDTH-1)/TILE_WIDTH,(SEQ_LEN+TILE_WIDTH-1)/TILE_WIDTH);
   transposeKernel<<<gbt2,bb>>>(d_txt_K,d_txt_KT,SEQ_LEN,D_MODEL);
 
   dim3 gb_ts((SEQ_LEN+TILE_WIDTH-1)/TILE_WIDTH,(SEQ_LEN+TILE_WIDTH-1)/TILE_WIDTH);
   tiledMatMulKernel<<<gb_ts,bb>>>(d_txt_Q,d_txt_KT,d_txt_S,SEQ_LEN,D_MODEL,SEQ_LEN);
   scaleKernel<<<(SEQ_LEN*SEQ_LEN+255)/256,256>>>(d_txt_S,SEQ_LEN*SEQ_LEN,
                                                  1.f/std::sqrt((float)D_HEAD));
   softmaxKernel<<<SEQ_LEN,256,2*sizeof(float)>>>(d_txt_S,SEQ_LEN,SEQ_LEN);
 
   tiledMatMulKernel<<<gb_ts,bb>>>(d_txt_S,d_txt_V,d_txt_A,SEQ_LEN,SEQ_LEN,D_MODEL);
   tiledMatMulKernel<<<gb_tx,bb>>>(d_txt_A,d_Wo_t,d_txt_P,SEQ_LEN,D_MODEL,D_MODEL);
 
   int elems=SEQ_LEN*D_MODEL;
   addInPlace<<<(elems+255)/256,256>>>(d_txt_P,d_emb_in,elems);
   layer_norm_kernel<<<SEQ_LEN,256,256*sizeof(float)>>>(d_txt_P,d_norm1,SEQ_LEN,D_MODEL,1e-5f);
 
   dim3 gb_ff1((D_FF+TILE_WIDTH-1)/TILE_WIDTH,(SEQ_LEN+TILE_WIDTH-1)/TILE_WIDTH);
   tiledMatMulKernel<<<gb_ff1,bb>>>(d_norm1,d_W1,d_ff1,SEQ_LEN,D_MODEL,D_FF);
   reluKernel<<<(SEQ_LEN*D_FF+255)/256,256>>>(d_ff1,SEQ_LEN*D_FF);
   dim3 gb_ff2((D_MODEL+TILE_WIDTH-1)/TILE_WIDTH,(SEQ_LEN+TILE_WIDTH-1)/TILE_WIDTH);
   tiledMatMulKernel<<<gb_ff2,bb>>>(d_ff1,d_W2,d_ff2,SEQ_LEN,D_FF,D_MODEL);
 
   addInPlace<<<(elems+255)/256,256>>>(d_ff2,d_norm1,elems);
   layer_norm_kernel<<<SEQ_LEN,256,256*sizeof(float)>>>(d_ff2,d_norm2,SEQ_LEN,D_MODEL,1e-5f);
 
   CUDA_CHECK(cudaMemcpy(d_txt_vec,d_norm2,D_MODEL*sizeof(float),cudaMemcpyDeviceToDevice));
   l2_norm_kernel<<<1,D_MODEL>>>(d_txt_vec);
 
   dot_kernel<<<1,D_MODEL>>>(d_img_vec,d_txt_vec,d_sim);
 
   CUDA_CHECK(cudaEventRecord(k1));
 
   // Device-to-host copy
   CUDA_CHECK(cudaEventRecord(d0));
   float sim_gpu;
   CUDA_CHECK(cudaMemcpy(&sim_gpu,d_sim,sizeof(float),cudaMemcpyDeviceToHost));
   CUDA_CHECK(cudaEventRecord(d1));
 
   CUDA_CHECK(cudaEventRecord(t1));
   CUDA_CHECK(cudaEventSynchronize(t1));
 
   // Timing
   float ms_h2d,ms_k,ms_d2h,ms_tot;
   CUDA_CHECK(cudaEventElapsedTime(&ms_h2d,h0,h1));
   CUDA_CHECK(cudaEventElapsedTime(&ms_k ,k0,k1));
   CUDA_CHECK(cudaEventElapsedTime(&ms_d2h,d0,d1));
   CUDA_CHECK(cudaEventElapsedTime(&ms_tot,t0,t1));
 
   std::cout<<"\nGPU cosine similarity: "<<sim_gpu<<"\n";
   std::cout<<"GPU timings (ms):\n"
            <<"  Host-to-Device copy time: : "<<ms_h2d<<"\n"
            <<"  Kernel execution time:    : "<<ms_k<<"\n"
            <<"  Device-to-Host copy time: : "<<ms_d2h<<"\n"
            <<"  Total GPU time:           : "<<ms_tot<<"\n";
 
   return 0;
 }
 