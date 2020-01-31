#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#include <stdio.h>
#include <math.h>
#define TILE_WIDTH 16
#define BLOCK_SIZE 1024

namespace mxnet
{
namespace op
{

__global__ void shared_memory_kernel(float *y,  //output
                               const float *x,  //input
                               const float *w,  //weight
                               const int B,
                               const int M,
                               const int C,
                               const int H,
                               const int W,
                               const int K)
{
    //const vars
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int W_grid = ceil(W_out/(TILE_WIDTH*1.0));
    const int tk1 = TILE_WIDTH+K-1;
    // shared memory vars
    extern __shared__ float shared[];
    float * s1 = &shared[0];
    float * s2 = &shared[(tk1)*(tk1)];
    //thread constants
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int h0 = (blockIdx.z/W_grid)*TILE_WIDTH;
    int w0 = (blockIdx.z%W_grid)*TILE_WIDTH;
    int h1 = h0 + tx;
    int w1 = w0 + ty;
    //calculation var
    float acc = 0.0;
    //loop vars
    int i,j,c,p,q;
    //macros
    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define w4d(i3, i2, i1, i0) w[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    //begin calculation
    //all inputs
    for(c=0; c<C; c++){
      if(tx<K && ty < K){
        //save in shared mem
        s2[tx*K+ty] = w4d(by, c, tx, ty);
      }
      __syncthreads();
      for(i = h1; i<h0+tk1; i+=TILE_WIDTH){
        for(j = w1; j<w0+tk1; j+=TILE_WIDTH){
          //save in shared mem
          s1[(i - h0) * (tk1) + j - w0] = x4d(bx, c, i, j);
        }
      }
      __syncthreads();
      //filter
      for(p=0; p<K; p++){
        for(q = 0; q<K; q++){
          acc+=s1[(tx + p) * (tk1) + ty + q] * s2[p*K + q];
        }
      }
      __syncthreads();
    }
    if(h1 < H_out && w1 < W_out){
      //set output
      y4d(bx,by,h1,w1) = acc;
    }
    #undef y4d
    #undef x4d
    #undef k4d
}

template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y,
                         const mshadow::Tensor<gpu, 4, float> &x,
                         const mshadow::Tensor<gpu, 4, float> &w)
{

    // CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";

    // Extract the tensor dimensions into B,M,C,H,W,K
    const int B = x.shape_[0]; //B is number of images in batch
    const int M = y.shape_[1]; //M is number of output feature maps
    const int C = x.shape_[1]; //C is number of input feature maps
    const int H = x.shape_[2]; //H is height of input map image
    const int W = x.shape_[3]; //W is width of input map image
    const int K = w.shape_[3]; //K is height/width of each filter W[M,C, K, K]
        //host constants
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    int H_grid = ceil(H_out / (1.0*TILE_WIDTH));
    int W_grid = ceil(W_out / (1.0*TILE_WIDTH));
    const int Z = H_grid*W_grid;
        // Set the kernel dimensions
    dim3 dimGrid(B,M,Z);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    unsigned int tk1 = TILE_WIDTH+K-1;
    unsigned int size = sizeof(float)*(tk1*tk1+K*K);
    shared_memory_kernel<<<dimGrid, dimBlock, size>>>(y.dptr_, x.dptr_, w.dptr_, B,M,C,H,W,K);
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
}
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif
