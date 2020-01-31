#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define TILE_WIDTH 16
#define BLOCK_WIDTH 1024

#include <mxnet/base.h>


__constant__ float weight[8000];
namespace mxnet
{
namespace op
{

__global__ void unrolled_kernel(int H, int W, int K, int C, float* X_unrolled, float* X) {
  int tx = blockIdx.x * BLOCK_WIDTH + threadIdx.x;
  int H_out = H-K+1;
  int W_out = W-K+1;
  int W_unrolled_size = H_out*W_out;
  int tx_max = C*W_unrolled_size;

  if(tx<tx_max){
    //unrolling alg variables
    int x = tx/W_unrolled_size;
    int H_unrolled_size = x*K*K;
    int y = tx%W_unrolled_size;
    int h = y/W_out;
    int w = y%W_out;
    int w1 = h*W_out+w;
    //loop variables
    int i, j;
    //loop

    #define x2d(i1, i0) X[x*H*W + (h+i1) * W + w + i0]

    for(i = 0; i<K; i++){
      for(j = 0; j<K; j++){
        int temp = H_unrolled_size +i*K+ j;
        X_unrolled[temp * W_unrolled_size + w1] = x2d(i,j);
      }
    }
    #undef x2d
  }
}
__global__ void matrix_multiplcation_kernel(float * X_unrolled, float* out,
      int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns){
        //thread and index variables
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int index = blockIdx.x*TILE_WIDTH + tx;
        int indey = blockIdx.y*TILE_WIDTH + ty;
        //shared memory
        __shared__ float X[TILE_WIDTH][TILE_WIDTH];
        __shared__ float Y[TILE_WIDTH][TILE_WIDTH];

        float acc = 0.0;
        int i, j; //loop variables
        int I = ceil(numAColumns/(TILE_WIDTH*1.0));
        for (i = 0; i<I; i++) {
          if ((indey<numCRows) && ((i*TILE_WIDTH+tx)<numAColumns)) {
            X[ty][tx] = weight[indey*numAColumns + i*TILE_WIDTH + tx];
          }
          else X[ty][tx] = 0.0;
          if ((index<numCColumns) && ((i*TILE_WIDTH+ty)<numAColumns)) {
            Y[ty][tx] = X_unrolled[(i*TILE_WIDTH + ty) * numCColumns + index];
          }
          else Y[ty][tx] = 0.0;
          __syncthreads();
          for (j = 0; j < TILE_WIDTH; j++) {
            acc += X[ty][j] * Y[j][tx];
          }
          __syncthreads();
        }
        if (index<numCColumns && indey<numCRows) {
          out[indey*numCColumns+index] = acc;
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
//unrolled kernel var
    float * X_unrolled;
    cudaMalloc((void **)&X_unrolled, sizeof(float)*H_unrolled_size*W_unrolled_size);
    cudaMemcpyToSymbol(weight, w.dptr_, sizeof(float)*M*C*K*K);
    int i; //loop var
    for (i = 0; i < B; i++) {  //for each image
      //calculate kernel dimensionss
        int threads_needed = C * H_out * W_out;
        dim3 gridDim(ceil(threads_needed/(BLOCK_WIDTH*1.0)));
        dim3 blockDim(BLOCK_WIDTH);
        dim3 gridDim2(ceil(W_unrolled_size/(1.0 * TILE_WIDTH)), ceil(M/(1.0 * TILE_WIDTH)), 1);
        dim3 blockDim2(TILE_WIDTH, TILE_WIDTH , 1);
        float * X = x.dptr_ + (i*C*H*W);
        float * Y = y.dptr_ + (i*M*H_out*W_out);
        //unroll
        unrolled_kernel<<<gridDim, blockDim>>>(H, W, K, C, X_unrolled, X);
        // and multiply!
        matrix_multiplcation_kernel<<<gridDim2, blockDim2>>>(X_unrolled, Y, M, H_unrolled_size,
                                            H_unrolled_size, W_unrolled_size, M, W_unrolled_size);
    }

    cudaFree(X_unrolled);

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
