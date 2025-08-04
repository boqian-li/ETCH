#include <cmath>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <vector>

at::Tensor ball_query_cuda(int b, int n, int m, float radius, int nsample, 
     at::Tensor new_xyz,
     at::Tensor xyz, 
     at::Tensor idx); //idx(b, m, nsample)

std::vector<at::Tensor> anchor_query_cuda(
    at::Tensor sample_idx,        //bxnc
    at::Tensor grouped_indices,   //bxncxnn
    at::Tensor grouped_xyz,       //bxncxnnx3
    at::Tensor anchors,           //nax3
    at::Tensor kernel_points,     //ksx2
    at::Tensor anchor_weights,    //bxncxnaxksxnn
    int nq);

std::vector<at::Tensor> anchor_kp_query_cuda(
    at::Tensor sample_idx,        //bxnc
    at::Tensor grouped_indices,   //bxncxnn
    at::Tensor grouped_xyz,       //bxncxnnx3
    at::Tensor anchors,           //nax3
    at::Tensor kernel_points,     //ksx3
    at::Tensor anchor_neighbors,  //bxncxnaxann
    at::Tensor anchor_weights,    //bxncxnaxksxann
    int ann,
    int nq,
    float radius,
    float aperature);


/*
std::vector<at::Tensor> initial_anchor_query_cuda(
    at::Tensor centers, // [b, nc, 3]
    at::Tensor xyz, // [m, 3]
    at::Tensor kernel_points, // [ks, na, 3]
    at::Tensor anchor_weights, // [b, ks, nc, na]
    float radius,
    float sigma
    )
*/

std::vector<at::Tensor> initial_anchor_query_cuda(
    at::Tensor centers, // [b, 3, nc]
    at::Tensor xyz, // [m, 3]
    at::Tensor kernel_points, // [ks, na, 3]
    at::Tensor anchor_weights, // [b, ks, nc, na]
    at::Tensor anchor_ctn, // [b, ks, nc, na]
    float radius,
    float sigma);


at::Tensor furthest_point_sampling_cuda(
    at::Tensor source,      // bxnqx3
    at::Tensor temp,        // bxnq
    at::Tensor sampled_idx, // bxm
    int m);

// C++ interface
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


at::Tensor ball_query(
    at::Tensor new_xyz,  // (b,3,m)
    at::Tensor xyz,      // (b,3,n)
    const float radius,   
    const int nsample) {

    CHECK_INPUT(new_xyz);
    CHECK_INPUT(xyz);

    at::Tensor idx =
    torch::zeros({new_xyz.size(0), new_xyz.size(2), nsample},
             at::device(new_xyz.device()).dtype(at::ScalarType::Int));
    
    // output: idx(b, m, nsample)
    return ball_query_cuda(xyz.size(0), xyz.size(2), new_xyz.size(2), radius, nsample, new_xyz, xyz, idx);
}

std::vector<at::Tensor> anchor_query(
    at::Tensor sample_idx, //[b, p]
    at::Tensor grouped_indices,  //[b, p, nn]
    at::Tensor grouped_xyz, // [b, 3, p, nn]
    at::Tensor anchors, // [na, 3]
    at::Tensor kernel_points, // [ks, 2]
    const int nq) {

    CHECK_INPUT(sample_idx);
    CHECK_INPUT(grouped_indices);
    CHECK_INPUT(grouped_xyz);
    CHECK_INPUT(anchors);
    CHECK_INPUT(kernel_points);

    // [b, p, na, ks, nn]
    at::Tensor anchor_weights = torch::full({grouped_xyz.size(0), grouped_xyz.size(2), anchors.size(0), 
      kernel_points.size(0), grouped_indices.size(2)}, 1e6, at::device(grouped_xyz.device()).dtype(grouped_xyz.dtype()));

    return anchor_query_cuda(sample_idx, grouped_indices, grouped_xyz, anchors, 
      kernel_points, anchor_weights, nq);
}

std::vector<at::Tensor> anchor_kp_query(
    at::Tensor sample_idx, //[b, p]
    at::Tensor grouped_indices,  //[b, p, nn]
    at::Tensor grouped_xyz, // [b, 3, p, nn]
    at::Tensor anchors, // [na, 3]
    at::Tensor kernel_points, // [ks, 3]
    const int ann, const int nq, const float radius, const float aperature) {

    CHECK_INPUT(sample_idx);
    CHECK_INPUT(grouped_indices);
    CHECK_INPUT(grouped_xyz);
    CHECK_INPUT(anchors);
    CHECK_INPUT(kernel_points);

    // [b, p, na, ann]
    at::Tensor anchor_neighbors = torch::zeros({grouped_xyz.size(0), grouped_xyz.size(2), anchors.size(0), ann},
    at::device(grouped_xyz.device()).dtype(at::ScalarType::Int));

    // [b, p, na, ks, ann]
    at::Tensor anchor_weights = torch::zeros({grouped_xyz.size(0), grouped_xyz.size(2), anchors.size(0), 
      kernel_points.size(0), ann}, at::device(grouped_xyz.device()).dtype(grouped_xyz.dtype()));

    return anchor_kp_query_cuda(sample_idx, grouped_indices, grouped_xyz, anchors, 
      kernel_points, anchor_neighbors, 
      anchor_weights, ann, nq, radius, aperature);
}


std::vector<at::Tensor> initial_anchor_query(
    at::Tensor centers, //[b, 3, nc]
    at::Tensor xyz,  //[m, 3]
    at::Tensor kernel_points, // [ks, na, 3]
    const float radius, const float sigma) {

    CHECK_INPUT(centers);
    CHECK_INPUT(xyz);
    CHECK_INPUT(kernel_points);

    // b, ks, nc, na]
    at::Tensor anchor_weights = torch::zeros({centers.size(0), kernel_points.size(0), centers.size(2), 
      kernel_points.size(1)}, at::device(xyz.device()).dtype(xyz.dtype()));

    // b, ks, nc, na]
    at::Tensor anchor_ctn = torch::zeros({centers.size(0), kernel_points.size(0), centers.size(2), 
      kernel_points.size(1)}, at::device(xyz.device()).dtype(xyz.dtype()));

    return initial_anchor_query_cuda(centers, xyz,  kernel_points, 
				     anchor_weights, anchor_ctn, radius, sigma);
}

at::Tensor furthest_point_sampling(
    at::Tensor source_xyz, // [b, 3, p]
    const int m) {

    CHECK_INPUT(source_xyz);

    // [nb, nq]
    at::Tensor tmp = torch::full({source_xyz.size(0), source_xyz.size(2)}, 
      1e10, at::device(source_xyz.device()).dtype(source_xyz.dtype()));
    // [nb, m]
    at::Tensor sampled_idx = torch::zeros({source_xyz.size(0), m}, 
      at::device(source_xyz.device()).dtype(at::ScalarType::Int));

    return furthest_point_sampling_cuda(source_xyz, tmp, sampled_idx, m);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ball_query", &ball_query, "ball query (CUDA)");
  m.def("anchor_query", &anchor_query, "anchor query (CUDA)");
  m.def("initial_anchor_query", &initial_anchor_query, "initial anchor query with kernel points(CUDA)");
  m.def("furthest_point_sampling", &furthest_point_sampling, "furthest point sampling (CUDA)");
}
