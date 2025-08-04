#include <torch/torch.h>

#include <vector>

// CUDA forward declarations

at::Tensor spherical_conv_forward_cuda(
    at::Tensor anchor_neighbors, // [b, np, na, ks, ann] 
    at::Tensor anchor_weights,  // [b, np, na, ks, ann]
    at::Tensor support_point_feats, // [b, c_in, nq, na]
    at::Tensor anchor_feats // [b, c_in, ks, np, na]
    );

at::Tensor spherical_conv_backward_cuda(
    at::Tensor anchor_neighbors, // [b, np, na, ks, ann] 
    at::Tensor anchor_weights,  // [b, np, na, ks, ann] 
    at::Tensor grad_anchor_feats, // [b, c_in, ks, nc, na]
    at::Tensor grad_support_point_feats // [b, c_in, nq, na]
    );

at::Tensor intraspherical_conv_forward_cuda(
    at::Tensor anchor_neighbors,        // [na_out, ks, ann]  
    at::Tensor anchor_weights,          // [na_out, ks, ann]
    at::Tensor support_point_feats,     // [b, c_in, np, na_in]
    at::Tensor anchor_feats             // [b, c_in, ks, np, na_out]
    );

at::Tensor intraspherical_conv_backward_cuda(
    at::Tensor anchor_neighbors,        // [na_out, ks, ann]  
    at::Tensor anchor_weights,          // [na_out, ks, ann]
    at::Tensor grad_anchor_feats,       // [b, c_in, ks, np, na_out]
    at::Tensor grad_support_point_feats // [b, c_in, np, na_in]
    );

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor inter_zpconv_forward(
    at::Tensor anchor_neighbors, // [b, np, na, ks, ann] 
    at::Tensor anchor_weights,  // [b, np, na, ks, ann]
    at::Tensor support_point_feats // [b, c_in, nq, na]
    ) {
  CHECK_INPUT(anchor_neighbors);
  CHECK_INPUT(anchor_weights);
  CHECK_INPUT(support_point_feats);

  // output: [b, c_in, ks, np, na]
  at::Tensor anchor_feats = torch::zeros({anchor_neighbors.size(0), support_point_feats.size(1), anchor_neighbors.size(3), 
    anchor_neighbors.size(1), anchor_neighbors.size(2)}, 
    at::device(support_point_feats.device()).dtype(support_point_feats.dtype()));

  return spherical_conv_forward_cuda(anchor_neighbors, anchor_weights, support_point_feats, anchor_feats);
}

at::Tensor inter_zpconv_backward(
    at::Tensor anchor_neighbors, // [b, np, na, ks, ann] 
    at::Tensor anchor_weights,  // [b, np, na, ks, ann] 
    at::Tensor grad_anchor_feats, // [b, c_in, ks, np, na]
    const int npoint
    ) {
  CHECK_INPUT(anchor_neighbors);
  CHECK_INPUT(anchor_weights);
  CHECK_INPUT(grad_anchor_feats);

  // output: [b, c_in, nq, na]
  at::Tensor grad_support_point_feats = torch::zeros({anchor_neighbors.size(0), grad_anchor_feats.size(1),
    npoint, anchor_neighbors.size(2)}, 
    at::device(grad_anchor_feats.device()).dtype(grad_anchor_feats.dtype()));


  return spherical_conv_backward_cuda(anchor_neighbors, anchor_weights, grad_anchor_feats, grad_support_point_feats);
}

at::Tensor intra_zpconv_forward(
    at::Tensor anchor_neighbors, // [na_out, ann] 
    at::Tensor anchor_weights,  // [na_out, ks, ann]
    at::Tensor support_point_feats // [b, c_in, np, na_in]
    ) {
  CHECK_INPUT(anchor_neighbors);
  CHECK_INPUT(anchor_weights);
  CHECK_INPUT(support_point_feats);

  // output: [b, c_in, ks, np, na_out]
  at::Tensor anchor_feats = torch::zeros({support_point_feats.size(0), support_point_feats.size(1), anchor_weights.size(1), 
    support_point_feats.size(2), anchor_neighbors.size(0)}, 
    at::device(support_point_feats.device()).dtype(support_point_feats.dtype()));

  return intraspherical_conv_forward_cuda(anchor_neighbors, anchor_weights, support_point_feats, anchor_feats);
}

at::Tensor intra_zpconv_backward(
    at::Tensor anchor_neighbors, // [na_out, ann] 
    at::Tensor anchor_weights,  // [na_out, ks, ann] 
    at::Tensor grad_anchor_feats, // [b, c_in, ks, np, na_out]
    const int anchor_in
    ) {
  CHECK_INPUT(anchor_neighbors);
  CHECK_INPUT(anchor_weights);
  CHECK_INPUT(grad_anchor_feats);

  // output: [b, c_in, np, na_in]
  at::Tensor grad_support_point_feats = torch::zeros({grad_anchor_feats.size(0), grad_anchor_feats.size(1), 
    grad_anchor_feats.size(3), anchor_in}, 
    at::device(grad_anchor_feats.device()).dtype(grad_anchor_feats.dtype()));

  return intraspherical_conv_backward_cuda(anchor_neighbors, anchor_weights, grad_anchor_feats, grad_support_point_feats);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("inter_zpconv_forward", &inter_zpconv_forward, "inter conv forward (CUDA)");
  m.def("inter_zpconv_backward", &inter_zpconv_backward, "inter conv backward (CUDA)");
  m.def("intra_zpconv_forward", &intra_zpconv_forward, "intra conv forward (CUDA)");
  m.def("intra_zpconv_backward", &intra_zpconv_backward, "intra conv backward (CUDA)");
}