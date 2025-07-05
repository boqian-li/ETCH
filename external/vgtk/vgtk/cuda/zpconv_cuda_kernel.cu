#include <ATen/ATen.h>
// #include <torch/torch.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>


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


namespace
{


// InterZPConv Grouping
// np: n sample, nq: n point
template <typename scalar_t>
__global__ void spherical_conv_forward_cuda_kernel(
    const int* __restrict__ anchor_neighbors, // [b, np, na, ks, ann] 
    const scalar_t* __restrict__ anchor_weights, // [b, np, na, ks, ann]
    const scalar_t* __restrict__ support_point_feats, // [b, c_in, nq, na]
    scalar_t* anchor_feats, // [b, c_in, ks, np, na]
    int np,
    int nq,
    int na,
    int ks,
    int ann,
    int c_in) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x; // pn*na+an or ((pn*na+an)*ks+k)*ann + ni
    const int bn = blockIdx.y;

    // faster version
    const int ni = idx % ann;
    const int k = (idx/ann) % ks;
    const int an = (idx/(ann*ks)) % na;
    const int pn =  idx/(ann*ks*na);

    if (pn >= np || an >= na || k >= ks || ni >= ann)
      return;
    
    const int q_idx = bn*np*na*ks*ann+idx;
    const int feat_dim = bn*c_in*na;

    const int qn = anchor_neighbors[q_idx];
    const scalar_t neighbor_weight = anchor_weights[q_idx];
    const scalar_t* neighbor_feat = &support_point_feats[feat_dim*nq + qn*na+an];
    scalar_t* anchor_feat = &anchor_feats[feat_dim*np*ks + (k*np+pn)*na+an];
    
    const int stride_1 = ks*np*na;
    const int stride_2 = nq*na;
    for(int ci = 0; ci < c_in; ci++) {
      atomicAdd(anchor_feat, neighbor_feat[0] * neighbor_weight);
      anchor_feat += stride_1;
      neighbor_feat += stride_2;
    }

}


template <typename scalar_t>
__global__ void spherical_conv_backward_cuda_kernel(
    const int* __restrict__ anchor_neighbors, // [b, np, na, ks, ann]
    const scalar_t* __restrict__ anchor_weights, // [b, np, na, ks, ann]
    const scalar_t* __restrict__ grad_anchor_feats, // [b, c_in, ks, np, na]
    scalar_t* grad_support_point_feats, // [b, c_in, nq, na]
    int np,
    int nq,
    int na,
    int ks,
    int ann,
    int c_in) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x; // ((pn*na+an)*ks+k)*ann + ni
    const int bn = blockIdx.y;
      // faster version
    const int ni = idx % ann;
    const int k = (idx/ann)%ks;
    const int an = (idx/(ann*ks))%na;
    const int pn =  idx/(ann*ks*na);
    
    if (pn >= np || an >= na || k>=ks || ni >=ann)
      return;
    
    const int q_idx = bn*np*na*ks*ann+idx;
    const int feat_dim = bn*c_in*na;

    const int qn = anchor_neighbors[q_idx];
    const scalar_t neighbor_weight = anchor_weights[q_idx];
    const scalar_t* grad_anchor_feat = &grad_anchor_feats[feat_dim*np*ks + (k*np+pn)*na+an];
    scalar_t* grad_support_feat = &grad_support_point_feats[feat_dim*nq + qn*na+an];

    const int stride_1 = ks*np*na;
    const int stride_2 = nq*na;
    for(int ci =0; ci < c_in; ci++) {
      atomicAdd(grad_support_feat, grad_anchor_feat[0] * neighbor_weight);
      grad_anchor_feat += stride_1;
      grad_support_feat += stride_2;
    }

}

// IntraZPConv Grouping
template <typename scalar_t>
__global__ void intraspherical_conv_forward_cuda_kernel(
    const int* __restrict__ anchor_neighbors, // [na_out, ann] 
    const scalar_t* __restrict__ anchor_weights, // [na_out, ks, ann]
    const scalar_t* __restrict__ support_point_feats, // [b, c_in, np, na_in]
    scalar_t* anchor_feats, // [b, c_in, ks, np, na_out]
    int np,
    int na_in,
    int na_out,
    int ks,
    int ann,
    int c_in) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x; // ((pn*na_out+an)*ks+k)*ann + ni
    const int bn = blockIdx.y;

    // faster version
    const int ni = idx % ann;
    const int k = (idx/ann)%ks;
    const int an = (idx/(ann*ks))%na_out;
    const int pn =  idx/(ann*ks*na_out);
    if (pn >= np || an >= na_out || k>=ks || ni >=ann)
      return;
    
    const int qan = anchor_neighbors[an*ann + ni];
    const scalar_t neighbor_weight = anchor_weights[(an*ks + k)*ann + ni];
    const scalar_t* neighbor_feat = &support_point_feats[bn*c_in*np*na_in + pn*na_in+qan];
    scalar_t* anchor_feat = &anchor_feats[bn*c_in*ks*np*na_out + (k*np+pn)*na_out+an];
    
    const int stride_1 = ks*np*na_out;
    const int stride_2 = np*na_in;
    for(int ci = 0; ci < c_in; ci++) {
      atomicAdd(anchor_feat, neighbor_feat[0] * neighbor_weight);
      anchor_feat += stride_1;
      neighbor_feat += stride_2;
    }

}


template <typename scalar_t>
__global__ void intraspherical_conv_backward_cuda_kernel(
    const int* __restrict__ anchor_neighbors, // [na_out, ann] 
    const scalar_t* __restrict__ anchor_weights, // [na_out, ks, ann]
    const scalar_t* __restrict__ grad_anchor_feats, // [b, c_in, ks, np, na_out]
    scalar_t* grad_support_point_feats, // [b, c_in, np, na_in]
    int np,
    int na_in,
    int na_out,
    int ks,
    int ann,
    int c_in) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x; // ((pn*na_out+an)*ks+k)*ann + ni
    const int bn = blockIdx.y;
    // faster version
    const int ni = idx % ann;
    const int k = (idx/ann)%ks;
    const int an = (idx/(ann*ks))%na_out;
    const int pn =  idx/(ann*ks*na_out);
    
    if (pn >= np || an >= na_out || k>=ks || ni >=ann)
      return;
    
    const int qan = anchor_neighbors[an*ann + ni];
    const scalar_t neighbor_weight = anchor_weights[(an*ks + k)*ann + ni];
    const scalar_t* grad_anchor_feat = &grad_anchor_feats[bn*c_in*ks*np*na_out + (k*np+pn)*na_out+an];
    scalar_t* grad_support_feat = &grad_support_point_feats[bn*c_in*np*na_in + pn*na_in+qan];
    
    const int stride_1 = ks*np*na_out;
    const int stride_2 = np*na_in;
    for(int ci = 0; ci < c_in; ci++) {
      atomicAdd(grad_support_feat, grad_anchor_feat[0] * neighbor_weight);
      grad_anchor_feat += stride_1;
      grad_support_feat += stride_2;
    }
}

}


at::Tensor spherical_conv_forward_cuda(
    at::Tensor anchor_neighbors, // [b, np, na, ks, ann] 
    at::Tensor anchor_weights,  // [b, np, na, ks, ann]
    at::Tensor support_point_feats, // [b, c_in, nq, na]
    at::Tensor anchor_feats // [b, c_in, ks, np, na]
    ) {
  const auto batch_size = anchor_weights.size(0);
  const auto num_sample = anchor_weights.size(1);
  const auto num_anchor = anchor_weights.size(2);
  const auto kernel_size = anchor_weights.size(3);
  const auto num_anchornn = anchor_weights.size(4);
  const auto dim_in = support_point_feats.size(1);
  const auto num_support = support_point_feats.size(2);

  const int threads = 1024;
  const dim3 blocks((num_sample * num_anchor * kernel_size * num_anchornn + threads - 1) / threads, batch_size);
  //const dim3 blocks((num_sample * num_anchor + threads - 1) / threads, batch_size);

  AT_DISPATCH_FLOATING_TYPES(support_point_feats.type(), "spherical_conv_forward_cuda", ([&] {
    spherical_conv_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
        anchor_neighbors.data<int>(),
        anchor_weights.data<scalar_t>(),
        support_point_feats.data<scalar_t>(),
        anchor_feats.data<scalar_t>(),
        num_sample,
        num_support,
        num_anchor,
        kernel_size,
        num_anchornn,
        dim_in);
  }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) 
      printf("Error in forward_spherical_conv: %s\n", cudaGetErrorString(err));

  return anchor_feats;
}


at::Tensor spherical_conv_backward_cuda(
    at::Tensor anchor_neighbors, // [b, np, na, ks, ann] 
    at::Tensor anchor_weights,  // [b, np, na, ks, ann] 
    at::Tensor grad_anchor_feats, // [b, c_in, ks, np, na]
    at::Tensor grad_support_point_feats // [b, c_in, nq, na]
    ) {
  const auto batch_size = anchor_weights.size(0);
  const auto num_sample = anchor_weights.size(1);
  const auto num_anchor = anchor_weights.size(2);
  const auto kernel_size = anchor_weights.size(3);
  const auto num_anchornn = anchor_weights.size(4);
  const auto dim_in = grad_anchor_feats.size(1);
  const auto num_support = grad_support_point_feats.size(2);

  const int threads = 1024;
  const dim3 blocks((num_sample * num_anchor * kernel_size * num_anchornn + threads - 1) / threads, batch_size);
  //const dim3 blocks((num_sample * num_anchor + threads - 1) / threads, batch_size);

  AT_DISPATCH_FLOATING_TYPES(grad_anchor_feats.type(), "spherical_conv_backward_cuda", ([&] {
    spherical_conv_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
        anchor_neighbors.data<int>(),
        anchor_weights.data<scalar_t>(),
        grad_anchor_feats.data<scalar_t>(),
        grad_support_point_feats.data<scalar_t>(),
        num_sample,
        num_support,
        num_anchor,
        kernel_size,
        num_anchornn,
        dim_in);
  }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) 
      printf("Error in backward_spherical_conv: %s\n", cudaGetErrorString(err));

  return grad_support_point_feats;
}


at::Tensor intraspherical_conv_forward_cuda(
    at::Tensor anchor_neighbors,        // [na_out, ann]  
    at::Tensor anchor_weights,          // [na_out, ks, ann]
    at::Tensor support_point_feats,     // [b, c_in, np, na_in]
    at::Tensor anchor_feats             // [b, c_in, ks, np, na_out]
    ) {
  const auto batch_size = anchor_feats.size(0);  
  const auto dim_in = anchor_feats.size(1);
  const auto kernel_size = anchor_feats.size(2);
  const auto num_support = anchor_feats.size(3);
  const auto num_anchor_out = anchor_feats.size(4);
  const auto num_anchor_in = support_point_feats.size(3);
  const auto num_anchornn = anchor_weights.size(2);

  const int threads = 1024;
  const dim3 blocks((num_support * num_anchor_out * kernel_size * num_anchornn + threads - 1) / threads, batch_size);
  
  AT_DISPATCH_FLOATING_TYPES(support_point_feats.type(), "intraspherical_conv_forward_cuda", ([&] {
    intraspherical_conv_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
        anchor_neighbors.data<int>(),
        anchor_weights.data<scalar_t>(),
        support_point_feats.data<scalar_t>(),
        anchor_feats.data<scalar_t>(),
        num_support,
        num_anchor_in,
        num_anchor_out,
        kernel_size,
        num_anchornn,
        dim_in);
  }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) 
      printf("Error in forward_intraspherical_conv: %s\n", cudaGetErrorString(err));

  return anchor_feats;
}


at::Tensor intraspherical_conv_backward_cuda(
    at::Tensor anchor_neighbors,        // [na_out, ks, ann]  
    at::Tensor anchor_weights,          // [na_out, ks, ann]
    at::Tensor grad_anchor_feats,       // [b, c_in, ks, np, na_out]
    at::Tensor grad_support_point_feats // [b, c_in, np, na_in]
    ) {

  const auto batch_size = grad_anchor_feats.size(0);
  const auto num_support = grad_anchor_feats.size(3);
  const auto num_anchor_out = grad_anchor_feats.size(4);
  const auto kernel_size = grad_anchor_feats.size(2);
  const auto dim_in = grad_anchor_feats.size(1);
  const auto num_anchor_in = grad_support_point_feats.size(3);
  const auto num_anchornn = anchor_weights.size(2);

  const int threads = 1024;
  const dim3 blocks((num_support * num_anchor_out * kernel_size * num_anchornn + threads - 1) / threads, batch_size);
  
  AT_DISPATCH_FLOATING_TYPES(grad_anchor_feats.type(), "intraspherical_conv_backward_cuda", ([&] {
    intraspherical_conv_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
        anchor_neighbors.data<int>(),
        anchor_weights.data<scalar_t>(),
        grad_anchor_feats.data<scalar_t>(),
        grad_support_point_feats.data<scalar_t>(),
        num_support,
        num_anchor_in,
        num_anchor_out,
        kernel_size,
        num_anchornn,
        dim_in);
  }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) 
      printf("Error in backward_intraspherical_conv: %s\n", cudaGetErrorString(err));

  return grad_support_point_feats;
}