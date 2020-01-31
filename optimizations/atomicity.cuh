#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define TILE_WIDTH 16
#define SPLIT 2

#include <mxnet/base.h>


__constant__ float weight[8000];
namespace mxnet
{
namespace op
{

__global__ void kernel_fusion_kernel_with_atomics(float *y,  //output
                               const float *x,  //input
                               const int C,
                               const int H,
                               const int W,
                               const int K
                               const int W_out,
                               const int H_out,
      int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns){
        //thread and index variables
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int index = blockIdx.x*TILE_WIDTH + tx;
        int indey = blockIdx.y*TILE_WIDTH + ty;
        int indez = blockIdx.z/SPLIT;

        //shared memory
        __shared__ float X[TILE_WIDTH][TILE_WIDTH];
        __shared__ float Y[TILE_WIDTH][TILE_WIDTH];

        //instantiate registers here
        int row, col, t1, t2, t3, w, h, p, q;
        float acc = 0.0;
        int modz = blockIdx.z%SPLIT;
        int numBlocks = ceil(numAColumns/(TILE_WIDTH*1.0));
        int loop_start = modz*numBlocks/SPLIT;
        int loop_end = (modz+1)*numBlocks/SPLIT;
        //loop through
        for(int i = loop_start; i<loop_end; i++){
          row = i*TILE_WIDTH+ty;
          col = i*TILE_WIDTH+tx;
          if(indey<numCRows && col<numAColumns){
            X[ty][tx] = weight[indey*numAColumns+col];
          }
          else X[ty][tx] = 0.0
          if(index<numCColumns && row<numBRows){
            t1 = row*numCColumns+index;
            t2 = t1%numBColumns;
            t3 = t1/numBColumns;
            w = t2%W_out;
            h = t2/W_out;
            p = t3 / K % K;
            q = t3 % K;
            Y[ty][tx] = x[(indez)*(C*H*W)+(t3/K/K)*(H*W)+(h+p)*(W)+w+q];
          }
          else Y[ty][tx] = 0.0;
          __syncthreads();
          for(int j = 0; j<TILE_WIDTH; j++){
            acc += X[ty][j]*Y[j][tx];
          }
          __syncthreads();
        }
        //paramater checks and save to output
        if(index<numCColumns && indey<numCRows){
          atomicAdd(&y[(indez*numCRows*numCColumns)+(indey*numCColumns)+index], acc);
        }
}
/*
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    //CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";

    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...
    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];

//kernel sizing values
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int H_unrolled_size = C * K * K;
    int W_unrolled_size = H_out*W_out;
//constant memory copy
    cudaMemcpyToSymbol(weight, w.dptr_, sizeof(float)*M*C*K*K);

        dim3 gridDim(ceil(W_unrolled_size/(TILE_WIDTH*1.0)), ceil(M/(TILE_WIDTH*1.0)), B*SPLIT);
        dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
        kernel_fusion_kernel_with_atomics<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, C, H, W, K, W_out, H_out,
              M, H_unrolled_size, H_unrolled_size, W_unrolled_size, M, W_unrolled_size));
    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
}

/*
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif
