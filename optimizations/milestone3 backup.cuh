
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#define TILE_WIDTH 16
namespace mxnet
{
namespace op
{
    
__global__ void forward_kernel(float *y,  //output
                               const float *x,  //input
                               const float *w,  //weight
                               const int B, 
                               const int M,
                               const int C, 
                               const int H, 
                               const int W, 
                               const int K)
{

        const int H_out = H - K + 1;
        const int W_out = W - K + 1;
	const int W_grid = (W_out+TILE_WIDTH-1)/TILE_WIDTH;

        #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
        #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
        #define w4d(i3, i2, i1, i0) w[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
	
	const int n = blockIdx.z; 
  	const int m = blockIdx.y; 
  	const int x_h = (blockIdx.x / W_grid)*TILE_WIDTH + threadIdx.y;
  	const int x_w = (blockIdx.x % W_grid)*TILE_WIDTH + threadIdx.x;
  
  
  	int c,p,q;
  	if((x_h<H_out) && (x_w<W_out)){
      		float acc = 0;
      		y4d(n, m, x_h, x_w) = 0;
      		for(c = 0; c<C; c++){
        		for(p = 0; p<K; p++){
          			for(q = 0; q<K; q++){
            				acc += x4d(n, c, x_h + p, x_w + q) * w4d(m, c, p, q);
          			}
        		}
      		}
	y4d(n, m, x_h, x_w) = acc;

    	}

	#undef y4d
	#undef x4d
	#undef w4d
}

/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
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
  
  	  	

    // Set the kernel dimensions
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int W_grid = (W_out+TILE_WIDTH-1)/TILE_WIDTH;
    const int H_grid = (H_out+TILE_WIDTH-1)/TILE_WIDTH;
    
    dim3 gridDim(W_grid*H_grid, M, B);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);

    // Call the kernel
    forward_kernel<<<gridDim, blockDim, 0>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

    return;
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

