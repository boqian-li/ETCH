#include <cmath>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <vector>


at::Tensor gather_points_forward_cuda(
    at::Tensor support_points,
    at::Tensor grouped_indices,
    at::Tensor grouped_points
    ); 

at::Tensor gather_points_backward_cuda(
    at::Tensor grad_out,
    at::Tensor grouped_indices,
    at::Tensor grad_points
    );

// C++ interface
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


at::Tensor gather_points_forward(
    at::Tensor support_points, //[nb, c_in, nsupport]
    at::Tensor grouped_indices //[nb, nsample]
  ){


  CHECK_INPUT(support_points);
  CHECK_INPUT(grouped_indices);

  at::Tensor output = torch::zeros({support_points.size(0), support_points.size(1), grouped_indices.size(1)},
    at::device(support_points.device()).dtype(at::ScalarType::Float));

  // output: [nb, c_in, nsample]
  return gather_points_forward_cuda(support_points, grouped_indices, output);
}

at::Tensor gather_points_backward(
    at::Tensor grad_out,        //[nb, c_in, nsample]
    at::Tensor grouped_indices, //[nb, nsample]
    int npoint
  ){


  CHECK_INPUT(grad_out);
  CHECK_INPUT(grouped_indices);

  at::Tensor output = torch::zeros({grad_out.size(0), grad_out.size(1), npoint},
    at::device(grad_out.device()).dtype(grad_out.dtype()));

  // output: [nb, c_in, nsupport]
  return gather_points_backward_cuda(grad_out, grouped_indices, output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gather_points_forward", &gather_points_forward, "gathering forward (CUDA)");
  m.def("gather_points_backward", &gather_points_backward, "gathering backward (CUDA)");
}