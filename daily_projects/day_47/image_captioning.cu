// vlm_image_captioning.cu
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdlib>

// Error check
#define CUDA_CHECK(call)                                                   \
  do {                                                                      \
    cudaError_t err = call;                                                 \
    if (err != cudaSuccess) {                                               \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err)                \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;      \
      std::exit(EXIT_FAILURE);                                              \
    }                                                                       \
  } while(0)

// Parameters
#define IMAGE_HEIGHT   256
#define IMAGE_WIDTH    256
#define PATCH_SIZE     32
#define NUM_PATCHES_H  (IMAGE_HEIGHT/PATCH_SIZE)
#define NUM_PATCHES_W  (IMAGE_WIDTH /PATCH_SIZE)
#define NUM_PATCHES    (NUM_PATCHES_H*NUM_PATCHES_W)
#define PATCH_VEC_SIZE (PATCH_SIZE*PATCH_SIZE)

#define D_MODEL    256
#define H_HEADS     8
#define D_HEAD     (D_MODEL/H_HEADS)
#define D_FF       (4*D_MODEL)

#define MAX_TEXT_LEN 16
#define VOCAB_SIZE 10000

#define TILE_WIDTH 16

// CPU funcions

// Matmul A[M×K]×B[K×N]=C[M×N]
void cpu_matmul(const float *A,const float *B,float *C,
                int M,int K,int N){
  for(int i=0;i<M;i++) for(int j=0;j<N;j++){
    float s=0.f;
    for(int k=0;k<K;k++) s+=A[i*K+k]*B[k*N+j];
    C[i*N+j]=s;
  }
}

// Row-wise softmax
void cpu_softmax(float *m,int rows,int cols){
  for(int i=0;i<rows;i++){
    float *r=m+i*cols, M=r[0], sum=0.f;
    for(int j=1;j<cols;j++) if(r[j]>M) M=r[j];
    for(int j=0;j<cols;j++){ r[j]=expf(r[j]-M); sum+=r[j]; }
    for(int j=0;j<cols;j++) r[j]/=sum;
  }
}

// Patch embedding + sinusoidal positional embedding
void cpu_patch_embedding(const float *img,const float *W,float *out){
  for(int p=0;p<NUM_PATCHES;p++){
    int pr=p/NUM_PATCHES_W, pc=p%NUM_PATCHES_W;
    int sr=pr*PATCH_SIZE, sc=pc*PATCH_SIZE;
    for(int d=0;d<D_MODEL;d++){
      float s=0.f;
      for(int i=0;i<PATCH_SIZE;i++)for(int j=0;j<PATCH_SIZE;j++){
        s+=img[(sr+i)*IMAGE_WIDTH + sc+j]
         * W[((i*PATCH_SIZE+j)*D_MODEL)+d];
      }
      out[p*D_MODEL+d]=s;
    }
  }
  // add sinusoidal
  for(int p=0;p<NUM_PATCHES;p++)for(int d=0;d<D_MODEL;d++){
    float e=(2*(d/2))/(float)D_MODEL;
    float a=p/powf(10000.f,e);
    out[p*D_MODEL+d]+= (d%2?cosf(a):sinf(a));
  }
}

// Single‑layer transformer encoder
void cpu_encoder(const float *X,
                 const float *Wq,const float *Wk,const float *Wv,const float *Wo,
                 float *out){
  std::vector<float> Q(NUM_PATCHES*D_MODEL),
                     K(NUM_PATCHES*D_MODEL),
                     V(NUM_PATCHES*D_MODEL),
                     KT(D_MODEL*NUM_PATCHES),
                     scores(NUM_PATCHES*NUM_PATCHES),
                     attn(NUM_PATCHES*D_MODEL);
  cpu_matmul(X,Wq,Q.data(),NUM_PATCHES,D_MODEL,D_MODEL);
  cpu_matmul(X,Wk,K.data(),NUM_PATCHES,D_MODEL,D_MODEL);
  cpu_matmul(X,Wv,V.data(),NUM_PATCHES,D_MODEL,D_MODEL);
  // transpose K
  for(int i=0;i<NUM_PATCHES;i++)
    for(int d=0;d<D_MODEL;d++)
      KT[d*NUM_PATCHES+i]=K[i*D_MODEL+d];
  cpu_matmul(Q.data(),KT.data(),scores.data(),
             NUM_PATCHES,D_MODEL,NUM_PATCHES);
  float sf=1.f/sqrtf((float)D_MODEL);
  for(auto &v:scores)v*=sf;
  cpu_softmax(scores.data(),NUM_PATCHES,NUM_PATCHES);
  cpu_matmul(scores.data(),V.data(),attn.data(),
             NUM_PATCHES,NUM_PATCHES,D_MODEL);
  cpu_matmul(attn.data(),Wo,out,
             NUM_PATCHES,D_MODEL,D_MODEL);
}

// Cross‑modal attention (vis = NUM_PATCHES×D, txt = MAX_TEXT_LEN×D)
void cpu_cross_attn(const float *vis,const float *txt,
                    const float *Wq,const float *Wk,const float *Wv,const float *Wo,
                    float *out){
  std::vector<float> Q(NUM_PATCHES*D_MODEL),
                     K(MAX_TEXT_LEN*D_MODEL),
                     V(MAX_TEXT_LEN*D_MODEL),
                     KT(D_MODEL*MAX_TEXT_LEN),
                     scores(NUM_PATCHES*MAX_TEXT_LEN),
                     attn(NUM_PATCHES*D_MODEL);
  cpu_matmul(vis,Wq,Q.data(),NUM_PATCHES,D_MODEL,D_MODEL);
  cpu_matmul(txt,Wk,K.data(),MAX_TEXT_LEN,D_MODEL,D_MODEL);
  cpu_matmul(txt,Wv,V.data(),MAX_TEXT_LEN,D_MODEL,D_MODEL);
  for(int i=0;i<MAX_TEXT_LEN;i++)
    for(int d=0;d<D_MODEL;d++)
      KT[d*MAX_TEXT_LEN+i]=K[i*D_MODEL+d];
  cpu_matmul(Q.data(),KT.data(),scores.data(),
             NUM_PATCHES,D_MODEL,MAX_TEXT_LEN);
  float sf=1.f/sqrtf((float)D_HEAD);
  for(auto &v:scores)v*=sf;
  cpu_softmax(scores.data(),NUM_PATCHES,MAX_TEXT_LEN);
  cpu_matmul(scores.data(),V.data(),attn.data(),
             NUM_PATCHES,MAX_TEXT_LEN,D_MODEL);
  cpu_matmul(attn.data(),Wo,out,
             NUM_PATCHES,D_MODEL,D_MODEL);
}

// Single greedy decoder step
int cpu_decoder_step(const std::vector<float>& vis_feats,
                     const std::vector<int>& ids,
                     const float *tok_emb,
                     const float *Wq_s,const float *Wk_s,const float *Wv_s,const float *Wo_s,
                     const float *Wq_x,const float *Wk_x,const float *Wv_x,const float *Wo_x,
                     const float *W1,const float *W2,const float *Wout){
  int T=ids.size();
  std::vector<float> X(T*D_MODEL);
  for(int i=0;i<T;i++)
    for(int d=0;d<D_MODEL;d++)
      X[i*D_MODEL+d]=tok_emb[ids[i]*D_MODEL+d];

  // self‑attn
  std::vector<float> Q(T*D_MODEL),K(T*D_MODEL),V(T*D_MODEL),
                     KT(D_MODEL*T),scores(T*T),attn(T*D_MODEL),self_out(T*D_MODEL);
  cpu_matmul(X.data(),Wq_s,Q.data(),T,D_MODEL,D_MODEL);
  cpu_matmul(X.data(),Wk_s,K.data(),T,D_MODEL,D_MODEL);
  cpu_matmul(X.data(),Wv_s,V.data(),T,D_MODEL,D_MODEL);
  for(int i=0;i<T;i++)for(int d=0;d<D_MODEL;d++)
    KT[d*T+i]=K[i*D_MODEL+d];
  cpu_matmul(Q.data(),KT.data(),scores.data(),T,D_MODEL,T);
  float sf=1.f/sqrtf((float)D_HEAD);
  for(int i=0;i<T;i++)for(int j=0;j<T;j++){
    if(j>i) scores[i*T+j]=-1e9f;
    else     scores[i*T+j]*=sf;
  }
  cpu_softmax(scores.data(),T,T);
  cpu_matmul(scores.data(),V.data(),attn.data(),T,T,D_MODEL);
  cpu_matmul(attn.data(),Wo_s,self_out.data(),T,D_MODEL,D_MODEL);
  for(int i=0;i<T*D_MODEL;i++) X[i]+=self_out[i];

  // Cross‑attn for last token
  std::vector<float> Qc(D_MODEL),Kc(NUM_PATCHES*D_MODEL),Vc(NUM_PATCHES*D_MODEL),
                     KTc(D_MODEL*NUM_PATCHES),scores_c(NUM_PATCHES),attn_c(D_MODEL),cross_out(D_MODEL);
  for(int d=0;d<D_MODEL;d++) Qc[d]=X[(T-1)*D_MODEL+d];
  cpu_matmul(vis_feats.data(),Wk_x,Kc.data(),NUM_PATCHES,D_MODEL,D_MODEL);
  cpu_matmul(vis_feats.data(),Wv_x,Vc.data(),NUM_PATCHES,D_MODEL,D_MODEL);
  for(int i=0;i<NUM_PATCHES;i++)for(int d=0;d<D_MODEL;d++)
    KTc[d*NUM_PATCHES+i]=Kc[i*D_MODEL+d];
  cpu_matmul(Qc.data(),KTc.data(),scores_c.data(),1,D_MODEL,NUM_PATCHES);
  for(auto &v:scores_c)v*=sf;
  float Mv=scores_c[0], Se=0.f;
  for(int i=1;i<NUM_PATCHES;i++) if(scores_c[i]>Mv) Mv=scores_c[i];
  for(int i=0;i<NUM_PATCHES;i++){ scores_c[i]=expf(scores_c[i]-Mv); Se+=scores_c[i]; }
  for(int i=0;i<NUM_PATCHES;i++) scores_c[i]/=Se;
  for(int d=0;d<D_MODEL;d++){
    float s=0.f;
    for(int i=0;i<NUM_PATCHES;i++) s+=scores_c[i]*Vc[i*D_MODEL+d];
    attn_c[d]=s;
  }
  cpu_matmul(attn_c.data(),Wo_x,cross_out.data(),1,D_MODEL,D_MODEL);
  for(int d=0;d<D_MODEL;d++) X[(T-1)*D_MODEL+d]+=cross_out[d];

  // FFN
  std::vector<float> f1(D_FF),f2(D_MODEL);
  for(int i=0;i<D_FF;i++){
    float s=0.f;
    for(int d=0;d<D_MODEL;d++) s+=X[(T-1)*D_MODEL+d]*W1[d*D_FF+i];
    f1[i]=std::max(0.f,s);
  }
  for(int d=0;d<D_MODEL;d++){
    float s=0.f;
    for(int i=0;i<D_FF;i++) s+=f1[i]*W2[i*D_MODEL+d];
    X[(T-1)*D_MODEL+d]+=s;
  }

  // Project & argmax
  std::vector<float> logits(VOCAB_SIZE);
  for(int v=0;v<VOCAB_SIZE;v++){
    float s=0.f;
    for(int d=0;d<D_MODEL;d++) s+=X[(T-1)*D_MODEL+d]*Wout[d*VOCAB_SIZE+v];
    logits[v]=s;
  }
  return std::distance(logits.begin(),
                       std::max_element(logits.begin(),logits.end()));
}

// Generate full caption
std::vector<int> cpu_generate(const float *img,
                              const float *Wp,
                              const float *Wqe,const float *Wke,const float *Wve,const float *Woe,
                              const float *Wqc,const float *Wkc,const float *Wvc,const float *Woc,
                              const float *tok_emb,
                              const float *Wqs,const float *Wks,const float *Wvs,const float *Wos,
                              const float *Wqx,const float *Wkx,const float *Wvx,const float *Wox,
                              const float *W1,const float *W2,const float *Wout){
  std::vector<float> patch_emb(NUM_PATCHES*D_MODEL);
  cpu_patch_embedding(img,Wp,patch_emb.data());
  std::vector<float> vis_feats(NUM_PATCHES*D_MODEL);
  cpu_encoder(patch_emb.data(),
              Wqe,Wke,Wve,Woe,
              vis_feats.data());
  std::vector<float> dummy(MAX_TEXT_LEN*D_MODEL,0), cross_feats(NUM_PATCHES*D_MODEL);
  cpu_cross_attn(vis_feats.data(),dummy.data(),
                 Wqc,Wkc,Wvc,Woc,
                 cross_feats.data());
  std::vector<int> ids={1}; // <START>=1, <END>=2
  for(int t=1;t<MAX_TEXT_LEN && ids.back()!=2;t++){
    ids.push_back(cpu_decoder_step(
      cross_feats,ids,tok_emb,
      Wqs,Wks,Wvs,Wos,
      Wqx,Wkx,Wvx,Wox,
      W1,W2,Wout
    ));
  }
  return ids;
}

// GPU kernels

// Patch embedding
__global__
void patch_embedding_kernel(const float *img,const float *W,float *out){
  int p=blockIdx.x, tid=threadIdx.x;
  if(p>=NUM_PATCHES) return;
  __shared__ float sp[PATCH_VEC_SIZE];
  if(tid<PATCH_VEC_SIZE){
    int pr=p/NUM_PATCHES_W, pc=p%NUM_PATCHES_W;
    int sr=pr*PATCH_SIZE, sc=pc*PATCH_SIZE;
    int i=tid/PATCH_SIZE, j=tid%PATCH_SIZE;
    sp[tid]=img[(sr+i)*IMAGE_WIDTH + sc+j];
  }
  __syncthreads();
  if(tid<D_MODEL){
    float s=0.f;
    #pragma unroll
    for(int k=0;k<PATCH_VEC_SIZE;k++)
      s+=sp[k]*W[k*D_MODEL+tid];
    out[p*D_MODEL+tid]=s;
  }
}

// Positional embedding
__global__
void add_pos_emb_kernel(float *emb){
  int p=blockIdx.x, d=threadIdx.x;
  if(p<NUM_PATCHES&&d<D_MODEL){
    float e=(2*(d/2))/(float)D_MODEL;
    float a=p/powf(10000.f,e);
    emb[p*D_MODEL+d]+= (d%2?cosf(a):sinf(a));
  }
}

// Tiled matmul A[M×K]×B[K×N]→C[M×N]
__global__
void tiledMatMulKernel(const float *A,const float *B,float *C,
                       int M,int K,int N){
  __shared__ float sA[TILE_WIDTH][TILE_WIDTH],
                   sB[TILE_WIDTH][TILE_WIDTH];
  int row=blockIdx.y*TILE_WIDTH+threadIdx.y,
      col=blockIdx.x*TILE_WIDTH+threadIdx.x;
  float val=0.f;
  for(int t=0;t<(K+TILE_WIDTH-1)/TILE_WIDTH;t++){
    int Ac= t*TILE_WIDTH+threadIdx.x,
        Br= t*TILE_WIDTH+threadIdx.y;
    sA[threadIdx.y][threadIdx.x]=
      (row<M&&Ac<K)? A[row*K+Ac]:0.f;
    sB[threadIdx.y][threadIdx.x]=
      (Br<K&&col<N)? B[Br*N+col]:0.f;
    __syncthreads();
    for(int i=0;i<TILE_WIDTH;i++)
      val+= sA[threadIdx.y][i]*sB[i][threadIdx.x];
    __syncthreads();
  }
  if(row<M&&col<N) C[row*N+col]=val;
}

// Transpose A[rows×cols]→out[cols×rows]
__global__
void transposeKernel(const float *A,float *out,int rows,int cols){
  int r=blockIdx.y*blockDim.y+threadIdx.y,
      c=blockIdx.x*blockDim.x+threadIdx.x;
  if(r<rows&&c<cols) out[c*rows+r]=A[r*cols+c];
}

// Scale
__global__
void scaleKernel(float *m,int sz,float f){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<sz) m[i]*=f;
}

// Row‐wise softmax
__global__
void softmaxKernel(float *mat,int rows,int cols){
  int r=blockIdx.x;
  extern __shared__ float sh[];
  float *smax=sh, *ssum=sh+1;
  if(threadIdx.x==0){ 
    float M=mat[r*cols], S=0.f;
    for(int j=1;j<cols;j++) if(mat[r*cols+j]>M) M=mat[r*cols+j];
    *smax=M;
    for(int j=0;j<cols;j++){
      float e=expf(mat[r*cols+j]-M);
      mat[r*cols+j]=e;
      S+=e;
    }
    *ssum=S;
  }
  __syncthreads();
  // divide
  float S=*ssum;
  for(int j=threadIdx.x;j<cols;j+=blockDim.x)
    mat[r*cols+j]/=S;
}

// Causal mask for [M×M]
__global__
void causal_mask_kernel(float *s,int M){
  int i=blockIdx.y*blockDim.y+threadIdx.y,
      j=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<M&&j<M&&j>i) s[i*M+j]=-1e9f;
}

// In‑place add A[i]+=B[i]
__global__
void addInPlace(float *A,const float *B,int N){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<N) A[i]+=B[i];
}

// ReLU
__global__
void reluKernel(float *A,int N){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<N&&A[i]<0) A[i]=0;
}

// Extract one query vector from seq_emb[t‑1]
__global__
void extractQuery(const float *seq,int t,float *Qd){
  int d=blockIdx.x*blockDim.x+threadIdx.x;
  if(d<D_MODEL) Qd[d]=seq[t*D_MODEL+d];
}

// Argmax over VOCAB_SIZE
__global__
void argmax_kernel(const float *logits,int V,int *out){
  extern __shared__ float sdata[];
  float *sval=sdata;
  int *sidx=(int*)(sdata+blockDim.x);
  int tid=threadIdx.x;
  float bestv=-1e20f; int besti=0;
  for(int i=tid;i<V;i+=blockDim.x){
    float v=logits[i];
    if(v>bestv){ bestv=v; besti=i; }
  }
  sval[tid]=bestv; sidx[tid]=besti;
  __syncthreads();
  for(int s=blockDim.x/2;s>0;s>>=1){
    if(tid<s && sval[tid+s]>sval[tid]){
      sval[tid]=sval[tid+s];
      sidx[tid]=sidx[tid+s];
    }
    __syncthreads();
  }
  if(tid==0) out[0]=sidx[0];
}

// Embed token id→seq_emb[t*D_MODEL + d]
__global__
void embed_token_kernel(const int *ids,const float *tok_emb,
                        float *seq,int t){
  int d=blockIdx.x*blockDim.x+threadIdx.x;
  if(d<D_MODEL){
    int id=ids[t];
    seq[t*D_MODEL + d]= tok_emb[id*D_MODEL + d];
  }
}

// Write next_id[0] into gen_ids[t]
__global__
void write_id_kernel(int *gen_ids,const int *next_id,int t){
  if(threadIdx.x==0) gen_ids[t]=next_id[0];
}


int main(){
  std::srand(42);
  // Host buffers
  std::vector<float> h_img(IMAGE_HEIGHT*IMAGE_WIDTH),
                     h_Wp(PATCH_VEC_SIZE*D_MODEL),
                     h_Wqe(D_MODEL*D_MODEL),h_Wke(D_MODEL*D_MODEL),
                     h_Wve(D_MODEL*D_MODEL),h_Woe(D_MODEL*D_MODEL),
                     h_Wqc(D_MODEL*D_MODEL),h_Wkc(D_MODEL*D_MODEL),
                     h_Wvc(D_MODEL*D_MODEL),h_Woc(D_MODEL*D_MODEL),
                     h_tok_emb(VOCAB_SIZE*D_MODEL),
                     h_Wqs(D_MODEL*D_MODEL),h_Wks(D_MODEL*D_MODEL),
                     h_Wvs(D_MODEL*D_MODEL),h_Wos(D_MODEL*D_MODEL),
                     h_Wqx(D_MODEL*D_MODEL),h_Wkx(D_MODEL*D_MODEL),
                     h_Wvx(D_MODEL*D_MODEL),h_Wox(D_MODEL*D_MODEL),
                     h_W1(D_MODEL*D_FF),h_W2(D_FF*D_MODEL),
                     h_Wout(D_MODEL*VOCAB_SIZE);

  // random init
  for(auto &x:h_img)    x=rand()/float(RAND_MAX);
  for(auto &x:h_Wp)     x=rand()/float(RAND_MAX);
  for(auto &x:h_Wqe)    x=rand()/float(RAND_MAX);
  for(auto &x:h_Wke)    x=rand()/float(RAND_MAX);
  for(auto &x:h_Wve)    x=rand()/float(RAND_MAX);
  for(auto &x:h_Woe)    x=rand()/float(RAND_MAX);
  for(auto &x:h_Wqc)    x=rand()/float(RAND_MAX);
  for(auto &x:h_Wkc)    x=rand()/float(RAND_MAX);
  for(auto &x:h_Wvc)    x=rand()/float(RAND_MAX);
  for(auto &x:h_Woc)    x=rand()/float(RAND_MAX);
  for(auto &x:h_tok_emb)x=rand()/float(RAND_MAX);
  for(auto &x:h_Wqs)    x=rand()/float(RAND_MAX);
  for(auto &x:h_Wks)    x=rand()/float(RAND_MAX);
  for(auto &x:h_Wvs)    x=rand()/float(RAND_MAX);
  for(auto &x:h_Wos)    x=rand()/float(RAND_MAX);
  for(auto &x:h_Wqx)    x=rand()/float(RAND_MAX);
  for(auto &x:h_Wkx)    x=rand()/float(RAND_MAX);
  for(auto &x:h_Wvx)    x=rand()/float(RAND_MAX);
  for(auto &x:h_Wox)    x=rand()/float(RAND_MAX);
  for(auto &x:h_W1)     x=rand()/float(RAND_MAX);
  for(auto &x:h_W2)     x=rand()/float(RAND_MAX);
  for(auto &x:h_Wout)   x=rand()/float(RAND_MAX);

  // CPU inference
  auto t0=std::chrono::high_resolution_clock::now();
  auto cpu_ids=cpu_generate(
    h_img.data(),h_Wp.data(),
    h_Wqe.data(),h_Wke.data(),h_Wve.data(),h_Woe.data(),
    h_Wqc.data(),h_Wkc.data(),h_Wvc.data(),h_Woc.data(),
    h_tok_emb.data(),
    h_Wqs.data(),h_Wks.data(),h_Wvs.data(),h_Wos.data(),
    h_Wqx.data(),h_Wkx.data(),h_Wvx.data(),h_Wox.data(),
    h_W1.data(),h_W2.data(),h_Wout.data()
  );
  auto t1=std::chrono::high_resolution_clock::now();
  double cpu_ms=std::chrono::duration<double,std::milli>(t1-t0).count();
  std::cout<<"Total CPU inference time (ms): "<<cpu_ms<<" ms\n";

  // GPU allocations
  float
    *d_img,   *d_Wp,    *d_patch_emb,
    *d_Wqe,   *d_Wke,   *d_Wve,   *d_Woe,
    *d_Wqc,   *d_Wkc,   *d_Wvc,   *d_Woc,
    *d_tok_emb,
    *d_Wqs,   *d_Wks,   *d_Wvs,   *d_Wos,
    *d_Wqx,   *d_Wkx,   *d_Wvx,   *d_Wox,
    *d_W1,    *d_W2,    *d_Wout;
  int *d_gen_ids,*d_next_id;
  float *d_seq_emb,
        *d_enc_Q,*d_enc_K,*d_enc_V,*d_enc_KT,*d_enc_scores,*d_enc_attn,*d_enc_out,
        *d_dummy_txt,
        *d_cross_Q,*d_cross_K,*d_cross_V,*d_cross_KT,*d_cross_scores,*d_cross_attn,*d_cross_feats,
        *d_dec_Q,*d_dec_K,*d_dec_V,*d_dec_KT,*d_dec_scores,*d_dec_attn,*d_self_proj,
        *d_Qd,*d_Kdc,*d_Vdc,*d_KdcT,*d_scores_dc,*d_attn_dc,*d_proj_dc,*d_ffn1,*d_ffn2,*d_logits;

  #define ALLOC(ptr,TYPE,N) CUDA_CHECK(cudaMalloc(&ptr,(N)*sizeof(TYPE)))
  ALLOC(d_img,float,IMAGE_HEIGHT*IMAGE_WIDTH);
  ALLOC(d_Wp  ,float,PATCH_VEC_SIZE*D_MODEL);
  ALLOC(d_patch_emb,float,NUM_PATCHES*D_MODEL);

  ALLOC(d_Wqe,float,D_MODEL*D_MODEL);
  ALLOC(d_Wke,float,D_MODEL*D_MODEL);
  ALLOC(d_Wve,float,D_MODEL*D_MODEL);
  ALLOC(d_Woe,float,D_MODEL*D_MODEL);

  ALLOC(d_Wqc,float,D_MODEL*D_MODEL);
  ALLOC(d_Wkc,float,D_MODEL*D_MODEL);
  ALLOC(d_Wvc,float,D_MODEL*D_MODEL);
  ALLOC(d_Woc,float,D_MODEL*D_MODEL);

  ALLOC(d_tok_emb,float,VOCAB_SIZE*D_MODEL);

  ALLOC(d_Wqs,float,D_MODEL*D_MODEL);
  ALLOC(d_Wks,float,D_MODEL*D_MODEL);
  ALLOC(d_Wvs,float,D_MODEL*D_MODEL);
  ALLOC(d_Wos,float,D_MODEL*D_MODEL);

  ALLOC(d_Wqx,float,D_MODEL*D_MODEL);
  ALLOC(d_Wkx,float,D_MODEL*D_MODEL);
  ALLOC(d_Wvx,float,D_MODEL*D_MODEL);
  ALLOC(d_Wox,float,D_MODEL*D_MODEL);

  ALLOC(d_W1,float,D_MODEL*D_FF);
  ALLOC(d_W2,float,D_FF*D_MODEL);
  ALLOC(d_Wout,float,D_MODEL*VOCAB_SIZE);

  ALLOC(d_gen_ids,int,MAX_TEXT_LEN);
  ALLOC(d_next_id,int,1);
  ALLOC(d_seq_emb,float,MAX_TEXT_LEN*D_MODEL);

  // Encoder buffers
  ALLOC(d_enc_Q,float,NUM_PATCHES*D_MODEL);
  ALLOC(d_enc_K,float,NUM_PATCHES*D_MODEL);
  ALLOC(d_enc_V,float,NUM_PATCHES*D_MODEL);
  ALLOC(d_enc_KT,float,D_MODEL*NUM_PATCHES);
  ALLOC(d_enc_scores,float,NUM_PATCHES*NUM_PATCHES);
  ALLOC(d_enc_attn,float,NUM_PATCHES*D_MODEL);
  ALLOC(d_enc_out,float,NUM_PATCHES*D_MODEL);

  // Cross‑modal
  ALLOC(d_dummy_txt,float,MAX_TEXT_LEN*D_MODEL);
  ALLOC(d_cross_Q,float,NUM_PATCHES*D_MODEL);
  ALLOC(d_cross_K,float,MAX_TEXT_LEN*D_MODEL);
  ALLOC(d_cross_V,float,MAX_TEXT_LEN*D_MODEL);
  ALLOC(d_cross_KT,float,D_MODEL*MAX_TEXT_LEN);
  ALLOC(d_cross_scores,float,NUM_PATCHES*MAX_TEXT_LEN);
  ALLOC(d_cross_attn,float,NUM_PATCHES*D_MODEL);
  ALLOC(d_cross_feats,float,NUM_PATCHES*D_MODEL);

  // Decoder buffers
  ALLOC(d_dec_Q,float,MAX_TEXT_LEN*D_MODEL);
  ALLOC(d_dec_K,float,MAX_TEXT_LEN*D_MODEL);
  ALLOC(d_dec_V,float,MAX_TEXT_LEN*D_MODEL);
  ALLOC(d_dec_KT,float,D_MODEL*MAX_TEXT_LEN);
  ALLOC(d_dec_scores,float,MAX_TEXT_LEN*MAX_TEXT_LEN);
  ALLOC(d_dec_attn,float,MAX_TEXT_LEN*D_MODEL);
  ALLOC(d_self_proj,float,MAX_TEXT_LEN*D_MODEL);

  ALLOC(d_Qd,float,D_MODEL);
  ALLOC(d_Kdc,float,NUM_PATCHES*D_MODEL);
  ALLOC(d_Vdc,float,NUM_PATCHES*D_MODEL);
  ALLOC(d_KdcT,float,D_MODEL*NUM_PATCHES);
  ALLOC(d_scores_dc,float,NUM_PATCHES);
  ALLOC(d_attn_dc,float,D_MODEL);
  ALLOC(d_proj_dc,float,D_MODEL);
  ALLOC(d_ffn1,float,D_FF);
  ALLOC(d_ffn2,float,D_MODEL);
  ALLOC(d_logits,float,VOCAB_SIZE);

  // copy H2D
  cudaEvent_t e_h2d0,e_h2d1,e_k0,e_k1,e_d2h0,e_d2h1,e_tot0,e_tot1;
  CUDA_CHECK(cudaEventCreate(&e_h2d0));
  CUDA_CHECK(cudaEventCreate(&e_h2d1));
  CUDA_CHECK(cudaEventCreate(&e_k0));
  CUDA_CHECK(cudaEventCreate(&e_k1));
  CUDA_CHECK(cudaEventCreate(&e_d2h0));
  CUDA_CHECK(cudaEventCreate(&e_d2h1));
  CUDA_CHECK(cudaEventCreate(&e_tot0));
  CUDA_CHECK(cudaEventCreate(&e_tot1));

  CUDA_CHECK(cudaEventRecord(e_tot0));
  CUDA_CHECK(cudaEventRecord(e_h2d0));
  CUDA_CHECK(cudaMemcpy(d_img,    h_img.data(),    h_img.size()*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Wp,     h_Wp.data(),     h_Wp.size()*sizeof(float),    cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Wqe,    h_Wqe.data(),    h_Wqe.size()*sizeof(float),   cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Wke,    h_Wke.data(),    h_Wke.size()*sizeof(float),   cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Wve,    h_Wve.data(),    h_Wve.size()*sizeof(float),   cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Woe,    h_Woe.data(),    h_Woe.size()*sizeof(float),   cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Wqc,    h_Wqc.data(),    h_Wqc.size()*sizeof(float),   cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Wkc,    h_Wkc.data(),    h_Wkc.size()*sizeof(float),   cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Wvc,    h_Wvc.data(),    h_Wvc.size()*sizeof(float),   cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Woc,    h_Woc.data(),    h_Woc.size()*sizeof(float),   cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_tok_emb,h_tok_emb.data(),h_tok_emb.size()*sizeof(float),cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Wqs,    h_Wqs.data(),    h_Wqs.size()*sizeof(float),   cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Wks,    h_Wks.data(),    h_Wks.size()*sizeof(float),   cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Wvs,    h_Wvs.data(),    h_Wvs.size()*sizeof(float),   cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Wos,    h_Wos.data(),    h_Wos.size()*sizeof(float),   cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Wqx,    h_Wqx.data(),    h_Wqx.size()*sizeof(float),   cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Wkx,    h_Wkx.data(),    h_Wkx.size()*sizeof(float),   cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Wvx,    h_Wvx.data(),    h_Wvx.size()*sizeof(float),   cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Wox,    h_Wox.data(),    h_Wox.size()*sizeof(float),   cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_W1,     h_W1.data(),     h_W1.size()*sizeof(float),    cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_W2,     h_W2.data(),     h_W2.size()*sizeof(float),    cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Wout,   h_Wout.data(),   h_Wout.size()*sizeof(float),  cudaMemcpyHostToDevice));
  // zero dummy text
  CUDA_CHECK(cudaMemset(d_dummy_txt,0,MAX_TEXT_LEN*D_MODEL*sizeof(float)));
  CUDA_CHECK(cudaEventRecord(e_h2d1));

  // initialize generated ids[0]=1
  int start_token=1;
  CUDA_CHECK(cudaMemcpy(d_gen_ids,&start_token,sizeof(int),cudaMemcpyHostToDevice));
  // embed start
  {
    int blocks=(D_MODEL+255)/256;
    embed_token_kernel<<<blocks,256>>>(d_gen_ids,d_tok_emb,d_seq_emb,0);
  }

  // GPU kernels
  CUDA_CHECK(cudaEventRecord(e_k0));

  // patch embed + pos‑emb
  patch_embedding_kernel<<<NUM_PATCHES,D_MODEL>>>(d_img,d_Wp,d_patch_emb);
  add_pos_emb_kernel<<<NUM_PATCHES,D_MODEL>>>(d_patch_emb);

  // encoder
  {
    dim3 gb((D_MODEL+TILE_WIDTH-1)/TILE_WIDTH,
            (NUM_PATCHES+TILE_WIDTH-1)/TILE_WIDTH),
         bb(TILE_WIDTH,TILE_WIDTH);
    // Q K V
    tiledMatMulKernel<<<gb,bb>>>(d_patch_emb,d_Wqe,d_enc_Q,NUM_PATCHES,D_MODEL,D_MODEL);
    tiledMatMulKernel<<<gb,bb>>>(d_patch_emb,d_Wke,d_enc_K,NUM_PATCHES,D_MODEL,D_MODEL);
    tiledMatMulKernel<<<gb,bb>>>(d_patch_emb,d_Wve,d_enc_V,NUM_PATCHES,D_MODEL,D_MODEL);
    // Kᵀ
    dim3 tb((TILE_WIDTH),(TILE_WIDTH));
    dim3 tg((D_MODEL+TILE_WIDTH-1)/TILE_WIDTH,
            (NUM_PATCHES+TILE_WIDTH-1)/TILE_WIDTH);
    transposeKernel<<<tg,tb>>>(d_enc_K,d_enc_KT,NUM_PATCHES,D_MODEL);
    // scores = Q×Kᵀ
    dim3 gb2((NUM_PATCHES+TILE_WIDTH-1)/TILE_WIDTH,
             (NUM_PATCHES+TILE_WIDTH-1)/TILE_WIDTH);
    tiledMatMulKernel<<<gb2,bb>>>(d_enc_Q,d_enc_KT,d_enc_scores,
                                  NUM_PATCHES,D_MODEL,NUM_PATCHES);
    // scale + softmax
    int sc_sz=NUM_PATCHES*NUM_PATCHES;
    int sc_blk=(sc_sz+255)/256;
    scaleKernel<<<sc_blk,256>>>(d_enc_scores,sc_sz,1.f/sqrtf((float)D_MODEL));
    softmaxKernel<<<NUM_PATCHES,256,2*sizeof(float)>>>(d_enc_scores,
                                                       NUM_PATCHES,
                                                       NUM_PATCHES);
    // attn = scores×V
    tiledMatMulKernel<<<gb2,bb>>>(d_enc_scores,d_enc_V,d_enc_attn,
                                  NUM_PATCHES,NUM_PATCHES,D_MODEL);
    // out = attn×Wo
    tiledMatMulKernel<<<gb,bb>>>(d_enc_attn,d_Woe,d_enc_out,
                                  NUM_PATCHES,D_MODEL,D_MODEL);
  }

  // cross‑modal attention on dummy text
  {
    dim3 gb((D_MODEL+TILE_WIDTH-1)/TILE_WIDTH,
            (NUM_PATCHES+TILE_WIDTH-1)/TILE_WIDTH),
         bb(TILE_WIDTH,TILE_WIDTH);
    // Qc = enc_out × Wqc
    tiledMatMulKernel<<<gb,bb>>>(d_enc_out,d_Wqc,d_cross_Q,
                                  NUM_PATCHES,D_MODEL,D_MODEL);
    // Kc = dummy_txt × Wkc
    dim3 gb_t((D_MODEL+TILE_WIDTH-1)/TILE_WIDTH,
              (MAX_TEXT_LEN+TILE_WIDTH-1)/TILE_WIDTH);
    tiledMatMulKernel<<<gb_t,bb>>>(d_dummy_txt,d_Wkc,d_cross_K,
                                  MAX_TEXT_LEN,D_MODEL,D_MODEL);
    // Vc
    tiledMatMulKernel<<<gb_t,bb>>>(d_dummy_txt,d_Wvc,d_cross_V,
                                  MAX_TEXT_LEN,D_MODEL,D_MODEL);
    // Kcᵀ
    dim3 tg((D_MODEL+TILE_WIDTH-1)/TILE_WIDTH,
            (MAX_TEXT_LEN+TILE_WIDTH-1)/TILE_WIDTH);
    transposeKernel<<<tg,bb>>>(d_cross_K,d_cross_KT,MAX_TEXT_LEN,D_MODEL);
    // scores = Qc×Kcᵀ
    dim3 gb_s((NUM_PATCHES+TILE_WIDTH-1)/TILE_WIDTH,
              (MAX_TEXT_LEN+TILE_WIDTH-1)/TILE_WIDTH);
    tiledMatMulKernel<<<gb_s,bb>>>(d_cross_Q,d_cross_KT,d_cross_scores,
                                   NUM_PATCHES,D_MODEL,MAX_TEXT_LEN);
    // scale + softmax
    scaleKernel<<<(NUM_PATCHES*MAX_TEXT_LEN+255)/256,256>>>(d_cross_scores,
                                                            NUM_PATCHES*MAX_TEXT_LEN,
                                                            1.f/sqrtf((float)D_HEAD));
    softmaxKernel<<<NUM_PATCHES,256,2*sizeof(float)>>>(d_cross_scores,
                                                       NUM_PATCHES,
                                                       MAX_TEXT_LEN);
    // attn = scores×Vc
    dim3 gb_as((NUM_PATCHES+TILE_WIDTH-1)/TILE_WIDTH,
               (D_MODEL+TILE_WIDTH-1)/TILE_WIDTH);
    tiledMatMulKernel<<<gb_as,bb>>>(d_cross_scores,d_cross_V,d_cross_attn,
                                    NUM_PATCHES,MAX_TEXT_LEN,D_MODEL);
    // feat = attn×Wo_c
    tiledMatMulKernel<<<gb,bb>>>(d_cross_attn,d_Woc,d_cross_feats,
                                 NUM_PATCHES,D_MODEL,D_MODEL);
  }

  // iterative decode on GPU
  for(int t=1;t<MAX_TEXT_LEN;t++){
    // a) self‑attn on seq_emb[0:t]
    {
      int M=t;
      dim3 gb((D_MODEL+TILE_WIDTH-1)/TILE_WIDTH,
              (M+TILE_WIDTH-1)/TILE_WIDTH),
           bb(TILE_WIDTH,TILE_WIDTH);
      // Q,K,V
      tiledMatMulKernel<<<gb,bb>>>(d_seq_emb,d_Wqs,d_dec_Q,
                                   M,D_MODEL,D_MODEL);
      tiledMatMulKernel<<<gb,bb>>>(d_seq_emb,d_Wks,d_dec_K,
                                   M,D_MODEL,D_MODEL);
      tiledMatMulKernel<<<gb,bb>>>(d_seq_emb,d_Wvs,d_dec_V,
                                   M,D_MODEL,D_MODEL);
      // Kᵀ
      dim3 tg((D_MODEL+TILE_WIDTH-1)/TILE_WIDTH,
              (M+TILE_WIDTH-1)/TILE_WIDTH);
      transposeKernel<<<tg,bb>>>(d_dec_K,d_dec_KT,M,D_MODEL);
      // scores
      dim3 gb2((M+TILE_WIDTH-1)/TILE_WIDTH,
               (M+TILE_WIDTH-1)/TILE_WIDTH);
      tiledMatMulKernel<<<gb2,bb>>>(d_dec_Q,d_dec_KT,d_dec_scores,
                                    M,D_MODEL,M);
      // causal mask
      dim3 mg((M+TILE_WIDTH-1)/TILE_WIDTH,
              (M+TILE_WIDTH-1)/TILE_WIDTH);
      causal_mask_kernel<<<mg,bb>>>(d_dec_scores,M);
      // scale + softmax
      scaleKernel<<<(M*M+255)/256,256>>>(d_dec_scores,M*M,1.f/sqrtf((float)D_HEAD));
      softmaxKernel<<<M,256,2*sizeof(float)>>>(d_dec_scores,M,M);
      // attn
      tiledMatMulKernel<<<gb2,bb>>>(d_dec_scores,d_dec_V,d_dec_attn,
                                    M,M,D_MODEL);
      // project
      tiledMatMulKernel<<<gb,bb>>>(d_dec_attn,d_Wos,d_self_proj,
                                   M,D_MODEL,D_MODEL);
      // add to seq_emb
      int tot=M*D_MODEL;
      addInPlace<<<(tot+255)/256,256>>>(d_seq_emb,d_self_proj,tot);
    }

    // b) cross‑attn for last token
    {
      // Qd = seq_emb[t‑1]×Wqx  (M=1)
      dim3 gb((D_MODEL+TILE_WIDTH-1)/TILE_WIDTH,1),
           bb(TILE_WIDTH,TILE_WIDTH);
      // extract then matmul
      extractQuery<<<(D_MODEL+255)/256,256>>>(d_seq_emb,t-1,d_Qd);
      tiledMatMulKernel<<<gb,bb>>>(d_Qd,d_Wqx,d_proj_dc,1,D_MODEL,D_MODEL);

      // Kc,Vc from cross_feats
      tiledMatMulKernel<<<gb,bb>>>(d_cross_feats,d_Wkx,d_Kdc,
                                   NUM_PATCHES,D_MODEL,D_MODEL);
      tiledMatMulKernel<<<gb,bb>>>(d_cross_feats,d_Wvx,d_Vdc,
                                   NUM_PATCHES,D_MODEL,D_MODEL);
      // Kcᵀ
      dim3 tg((D_MODEL+TILE_WIDTH-1)/TILE_WIDTH,
              (NUM_PATCHES+TILE_WIDTH-1)/TILE_WIDTH);
      transposeKernel<<<tg,bb>>>(d_Kdc,d_KdcT,NUM_PATCHES,D_MODEL);
      // scores_dc = Qd×Kcᵀ
      dim3 gs((NUM_PATCHES+TILE_WIDTH-1)/TILE_WIDTH,1);
      tiledMatMulKernel<<<gs,bb>>>(d_proj_dc,d_KdcT,d_scores_dc,1,D_MODEL,NUM_PATCHES);
      // scale+softmax
      scaleKernel<<<(NUM_PATCHES+255)/256,256>>>(d_scores_dc,NUM_PATCHES,1.f/sqrtf((float)D_HEAD));
      softmaxKernel<<<1,256,2*sizeof(float)>>>(d_scores_dc,1,NUM_PATCHES);
      // attn_dc = scores_dc×Vdc
      dim3 gb2((D_MODEL+TILE_WIDTH-1)/TILE_WIDTH,(1+TILE_WIDTH-1)/TILE_WIDTH);
      tiledMatMulKernel<<<gb2,bb>>>(d_scores_dc,d_Vdc,d_attn_dc,
                                    1,NUM_PATCHES,D_MODEL);
      // proj
      tiledMatMulKernel<<<gb2,bb>>>(d_attn_dc,d_Wox,d_proj_dc,
                                    1,D_MODEL,D_MODEL);
      // add to seq_emb[t‑1]
      addInPlace<<<(D_MODEL+255)/256,256>>>(d_seq_emb+(t-1)*D_MODEL,d_proj_dc,D_MODEL);
    }

    // c) FFN on last token
    {
      // ffn1
      dim3 gb((D_FF+TILE_WIDTH-1)/TILE_WIDTH,1);
      dim3 bb(TILE_WIDTH,TILE_WIDTH);
      tiledMatMulKernel<<<gb,bb>>>(d_seq_emb+(t-1)*D_MODEL,d_W1,
                                   d_ffn1,1,D_MODEL,D_FF);
      reluKernel<<<(D_FF+255)/256,256>>>(d_ffn1,D_FF);
      // ffn2
      tiledMatMulKernel<<<(D_MODEL+TILE_WIDTH-1)/TILE_WIDTH,bb>>>(d_ffn1,d_W2,
                                                                   d_ffn2,1,D_FF,D_MODEL);
      // add
      addInPlace<<<(D_MODEL+255)/256,256>>>(d_seq_emb+(t-1)*D_MODEL,d_ffn2,D_MODEL);
    }

    // d) logits & argmax
    {
      dim3 gb((VOCAB_SIZE+TILE_WIDTH-1)/TILE_WIDTH,1);
      dim3 bb(TILE_WIDTH,TILE_WIDTH);
      tiledMatMulKernel<<<gb,bb>>>(d_seq_emb+(t-1)*D_MODEL,d_Wout,
                                   d_logits,1,D_MODEL,VOCAB_SIZE);
      argmax_kernel<<<1,1024,(1024*(sizeof(float)+sizeof(int)))>>>(d_logits,
                                                                   VOCAB_SIZE,
                                                                   d_next_id);
      write_id_kernel<<<1,1>>>(d_gen_ids,d_next_id,t);
      // embed new token
      int blocks=(D_MODEL+255)/256;
      embed_token_kernel<<<blocks,256>>>(d_gen_ids,d_tok_emb,d_seq_emb,t);
    }
  }

  CUDA_CHECK(cudaEventRecord(e_k1));

  // copy back generated ids
  CUDA_CHECK(cudaEventRecord(e_d2h0));
  std::vector<int> h_ids(MAX_TEXT_LEN);
  CUDA_CHECK(cudaMemcpy(h_ids.data(),d_gen_ids,MAX_TEXT_LEN*sizeof(int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(e_d2h1));
  CUDA_CHECK(cudaEventRecord(e_tot1));
  CUDA_CHECK(cudaEventSynchronize(e_tot1));

  float t_h2d,t_kern,t_d2h,t_tot;
  CUDA_CHECK(cudaEventElapsedTime(&t_h2d,e_h2d0,e_h2d1));
  CUDA_CHECK(cudaEventElapsedTime(&t_kern,e_k0 ,e_k1 ));
  CUDA_CHECK(cudaEventElapsedTime(&t_d2h,e_d2h0,e_d2h1));
  CUDA_CHECK(cudaEventElapsedTime(&t_tot ,e_tot0,e_tot1));

  std::cout<<"GPU inference timings (ms):\n"
           <<"  Host-to-Device copy time: "<<t_h2d<<"\n"
           <<"  Kernel execution time:    "<<t_kern<<"\n"
           <<"  Device-to-Host copy time: "<<t_d2h<<"\n"
           <<"  Total GPU time:           "<<t_tot<<"\n";

  return 0;
}
