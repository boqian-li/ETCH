#include <ATen/ATen.h>
// #include <torch/torch.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <stdio.h>
#include <stdlib.h>


#define TOTAL_THREADS 1024

// for the older gpus atomicAdd with double arguments does not exist
#if  __CUDA_ARCH__ < 600 and defined(__CUDA_ARCH__)
static __inline__ __device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN) } while (assumed != old);
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

namespace {


template <typename scalar_t> 
__device__ __forceinline__ void grbf(scalar_t &w, const scalar_t d, const float sig){
    w = exp(-0.5*d*d/(sig*sig));
}


// --------------------- Gathering and Grouping --------------------------

// input: points(b, c, n) idx(b, m)
// output: out(b, c, m)
template <typename scalar_t> 
__global__ void gather_points_forward_kernel(
    const scalar_t *__restrict__ points,
    const int *__restrict__ idx,
    scalar_t *__restrict__ out,
    int out_points,
    int num_channels,
    int num_indices
    ) {
    
    const int bn = blockIdx.y;
    //const int pn = blockIdx.x * blockDim.x + threadIdx.x;
    const int aidx = blockIdx.x * blockDim.x + threadIdx.x; // pn*nc + ci
    const int pn = aidx / num_channels;
    const int ci = aidx % num_channels;


    if (pn < num_indices && ci < num_channels){
        const int np = out_points;
        const int nc = num_channels;
        const int nm = num_indices;
        const int a = idx[bn*nm + pn];
        //printf("At kernel: bn %f, pn%f, nc %f, a %f \n", bn, pn, ci, a);  
        out[(bn*nc + ci) * nm + pn] = points[(bn*nc + ci)*np + a];
    
    }
}

// input: grad_out(b, c, m) idx(b, m)
// output: grad_points(b, c, n)
template <typename scalar_t> 
__global__ void gather_points_backward_kernel(
    const scalar_t *__restrict__ grad_out,
    const int *__restrict__ idx,
    scalar_t *__restrict__ grad_points,
    int out_points,
    int num_channels,
    int num_indices
    ) {

    const int bn = blockIdx.y;
    //const int pn = blockIdx.x * blockDim.x + threadIdx.x;
    const int aidx = blockIdx.x * blockDim.x + threadIdx.x; // pn*nc + ci
    const int pn = aidx / num_channels;
    const int ci = aidx % num_channels;

    if (pn < num_indices){
        const int np = out_points;
        const int nc = num_channels;
        const int nm = num_indices;

        const int a = idx[bn*nm + pn];
        atomicAdd(grad_points + (bn * nc + ci) * np + a, grad_out[(bn*nc + ci)*nm + pn]);
        
    }

}
}



at::Tensor gather_points_forward_cuda(
    at::Tensor support_points,
    at::Tensor grouped_indices,
    at::Tensor grouped_points
  ) {
    const auto batch_size = grouped_indices.size(0);
    const auto num_samples = grouped_indices.size(1);
    const auto num_support = support_points.size(2);
    const auto num_channel = support_points.size(1);
    
    // blocks, threads
    const int threads = 1024;
    const dim3 blocks((num_samples*num_channel + threads - 1) / threads, batch_size);

    AT_DISPATCH_FLOATING_TYPES(support_points.type(), "gather_points_forward_cuda", ([&] {
      gather_points_forward_kernel<scalar_t><<<blocks, threads>>>(
          support_points.data<scalar_t>(),
          grouped_indices.data<int>(),
          grouped_points.data<scalar_t>(),
          num_support,
          num_channel,
          num_samples);
    }));

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) 
            printf("Error: %s\n", cudaGetErrorString(err));

    return grouped_points;
}



at::Tensor gather_points_backward_cuda(
    at::Tensor grad_out,
    at::Tensor grouped_indices,
    at::Tensor grad_points
  ){
    const auto batch_size = grad_out.size(0);
    const auto num_samples = grad_out.size(2);
    const auto num_support = grad_points.size(2);
    const auto num_channel = grad_out.size(1);
    
    // blocks, threads
    const int threads = 1024;
    const dim3 blocks((num_samples*num_channel + threads - 1) / threads, batch_size);

    AT_DISPATCH_FLOATING_TYPES(grad_out.type(), "gather_points_backward_cuda", ([&] {
      gather_points_backward_kernel<scalar_t><<<blocks, threads>>>(
          grad_out.data<scalar_t>(),
          grouped_indices.data<int>(),
          grad_points.data<scalar_t>(),
          num_support,
          num_channel,
          num_samples);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error: %s\n", cudaGetErrorString(err));

    return grad_points;
}
