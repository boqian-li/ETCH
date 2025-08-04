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

inline int opt_n_threads(int work_size) {
    const int pow_2 = std::log(static_cast<double>(work_size)) / std::log(2.0);

    return max(min(1 << pow_2, TOTAL_THREADS), 1);
}

inline dim3 opt_block_config(int x, int y) {
    const int x_threads = opt_n_threads(x);
    const int y_threads =
        max(min(opt_n_threads(y), TOTAL_THREADS / x_threads), 1);
    dim3 block_config(x_threads, y_threads, 1);

    return block_config;
}

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
// -------------------- ball query ------------------------------

// input: new_xyz(b, m, 3) xyz(b, n, 3) -> output: idx(b, m, nsample)
// note: if samples are lacking, the rest is filled with shadow point indices

template <typename scalar_t>
__global__ void ball_query_cuda_kernel(int b, int n, int m, float radius,
    int nsample,
    const scalar_t *__restrict__ new_xyz,
    const scalar_t *__restrict__ xyz,
    int *__restrict__ idx) {

    int batch_index = blockIdx.x;
    xyz += batch_index * 3 * n;
    new_xyz += batch_index * 3 * m;
    idx += m * nsample * batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;

    scalar_t radius2 = radius * radius;
    for (int j = index; j < m; j += stride) {
        scalar_t new_x = new_xyz[0 * m + j];
        scalar_t new_y = new_xyz[1 * m + j];
        scalar_t new_z = new_xyz[2 * m + j];

        int cnt = 0;
        for (int k = 0; k < n && cnt < nsample; ++k) {
            scalar_t x = xyz[0 * n + k];
            scalar_t y = xyz[1 * n + k];
            scalar_t z = xyz[2 * n + k];
            scalar_t d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);
            if (d2 < radius2) {
                idx[j * nsample + cnt] = k;
                ++cnt;
            }
        }

        if (cnt < nsample - 1) {
            // if use repetitive sampling
            for(int k = 0; k + cnt < nsample; ++k) {
                idx[j*nsample + k + cnt] = idx[j*nsample + k];
            }

            // if use shadow points
            // for(; cnt < nsample; ++cnt) {

            //      idx[j*nsample + cnt] = n;
            // }
        }
    }
}


template <typename scalar_t>
__global__ void initial_anchor_query_cuda_kernel(
    const scalar_t* __restrict__ centers, //[b, 3, nc]
    const scalar_t* __restrict__ xyz, //[m, 3]
    const scalar_t* __restrict__ kernel_points, //[ks, na, 3]
    scalar_t* anchor_weights, //[b, ks, nc, na]
    scalar_t* anchor_ctn, //[b, ks, nc, na]
    int nb,
    int nc,
    int m,
    int na,
    int ks,
    float radius,
    float sigma)
{
    const int bn = blockIdx.y;
    //const int pn = blockIdx.x * blockDim.x + threadIdx.x;
    const int aidx = blockIdx.x * blockDim.x + threadIdx.x; // pm * ks + kn
    const int pm = aidx / ks;
    const int kn = aidx % ks;

    anchor_weights += bn * ks * nc * na;
    anchor_ctn += bn * ks * nc * na;
    centers += bn*3*nc;

    scalar_t x = xyz[3*pm];
    scalar_t y = xyz[3*pm+1];
    scalar_t z = xyz[3*pm+2];

    if (pm < m && kn < ks){
        for (int pn = 0; pn < nc; pn++) {
            scalar_t cx = centers[pn];
            scalar_t cy = centers[1*nc + pn];
            scalar_t cz = centers[2*nc + pn];
            scalar_t dist2cent = sqrt((cx - x)*(cx - x) + (cy - y)*(cy - y) + (cz - z)*(cz - z));

            if(dist2cent <= radius) {
                for (int an = 0; an < na; an++) {
                    scalar_t kx = kernel_points[(kn * na + an) * 3] + cx;
                    scalar_t ky = kernel_points[(kn * na + an) * 3 + 1] + cy;
                    scalar_t kz = kernel_points[(kn * na + an) * 3 + 2] + cz;
                    scalar_t dist2k = sqrt((kx - x)*(kx - x) + (ky - y)*(ky - y) + (kz - z)*(kz - z));
                    scalar_t weight = 1 - (dist2k * dist2k / sigma);
                    if (weight > 0.0){
                        atomicAdd(anchor_weights + (kn*nc + pn)*na + an , weight);
                    }
                    atomicAdd(anchor_ctn + (kn*nc + pn)*na + an , 1.0);
                }
            }
        }
    }
}

/*
    nc: # of center
    nn: # of point neighbors
    na: # of anchor directions
    ks: # of kernel points
    ann: # of anchor neighbors
*/
// note: 1) grouped_xyz must be local coordinates
//       2) returned anchor_weights are not normalized
//       3) if samples are lacking, the rest is filled with shadow point indices

template <typename scalar_t>
__global__ void anchor_query_cuda_kernel(
    const int* __restrict__ sample_idx, //[b, n]
    const int* __restrict__ grouped_indices, //[b, n, nn]
    const scalar_t* __restrict__ grouped_xyz, //[b, 3, n, nn]
    const scalar_t* __restrict__ anchors,  //[na, 3]
    const scalar_t* __restrict__ kernel_points, //[ks, 2] (w,h)
    scalar_t* anchor_weights, //[b, n, na, ks, nn]
    int np,
    int nn,
    int na,
    int nq,
    int ks){

    const int bi = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    grouped_indices += bi*np*nn;
    grouped_xyz += bi*3*np*nn;
    anchor_weights += bi*np*na*ks*nn;

    // really dirty and hacky don't use large kernel or anchor size
    __shared__ scalar_t s_kernels[TOTAL_THREADS];
    __shared__ scalar_t s_anchors[TOTAL_THREADS];
    if (tid < TOTAL_THREADS) {
        s_kernels[tid] = kernel_points[tid % (ks*2)];
        s_anchors[tid] = anchors[tid % (na*3)];
    }
    __syncthreads();

    // const int s_aid = tid % (stride / (3*na));
    // const int s_kid = tid % (stride / (2*ks));

    for (int idx = tid; idx < np*nn; idx += stride) {
        // printf("%d %d %d\n", tid, idx, stride);

        const int pi = idx / nn;
        const int ni = idx % nn;

        scalar_t x = grouped_xyz[0*np*nn+idx];
        scalar_t y = grouped_xyz[1*np*nn+idx];
        scalar_t z = grouped_xyz[2*np*nn+idx];
        scalar_t norm = sqrt(x*x + y*y + z*z) + 1e-6;

        for (int ai = 0; ai < na; ai++) {
            scalar_t theta = acos((x * s_anchors[ai*3+0] +
                                   y * s_anchors[ai*3+1] +
                                   z * s_anchors[ai*3+2]) / norm);
            // scalar_t theta = acos((x * s_anchors[s_aid*na*3+ai*3+0] +
            //                        y * s_anchors[s_aid*na*3+ai*3+1] +
            //                        z * s_anchors[s_aid*na*3+ai*3+2]) / norm);
            // if (theta != theta)
                // printf("%d, %f, %f, %f, %f\n", s_aid*na*3+ai*3+0, s_anchors[s_aid*na*3+ai*3+0], s_anchors[s_aid*na*3+ai*3+1], s_anchors[s_aid*na*3+ai*3+2], s_anchors[ai*3+1]);
            // scalar_t theta = 0;
            for (int ki = 0; ki < ks; ki++) {
                // scalar_t kw = s_kernels[s_kid*ks*2+ki*2+0];
                // scalar_t kh = s_kernels[s_kid*ks*2+ki*2+1];
                scalar_t kw = kernel_points[ki*2+0];
                scalar_t kh = kernel_points[ki*2+1];
                // scalar_t kw = 0;
                // scalar_t kh = 0;
                anchor_weights[(((pi*na)+ai)*ks+ki)*nn+ni] = pow(kw-norm, 2) + pow((kh-theta)*norm, 2);
            }
        }

    }
}


/*
    nc: # of center
    nn: # of point neighbors
    na: # of anchor directions
    ks: # of kernel points
    ann: # of anchor neighbors
*/
// note: 1) grouped_xyz must be local coordinates
//       2) returned anchor_weights are not normalized
//       3) if samples are lacking, the rest is filled with shadow point indices

template <typename scalar_t>
__global__ void anchor_kp_query_cuda_kernel(
    const int* __restrict__ sample_idx, //[b, n]
    const int* __restrict__ grouped_indices, //[b, n, nn]
    const scalar_t* __restrict__ grouped_xyz, //[b, 3, n, nn]
    const scalar_t* __restrict__ anchors,  //[na, 3]
    const scalar_t* __restrict__ kernel_points, //[ks, 3] (x,y,z)
    int* anchor_neighbors, //[b, n, na, ann]
    scalar_t* anchor_weights, //[b, n, na, ks, ann]
    int nc,
    int nn,
    int na,
    int nq,
    int ks,
    int ann,
    float radius,
    float aperature
    ){

    const int bn = blockIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; // pn*na+an or (pn*na+an)*ks+k

    const int k = idx % ks;
    const int a = idx/ks % na;
    const int pn = idx/(ks*na);
    if (pn >= nc || a >= na || k >= ks)
        return;

    grouped_xyz += (bn*3*nc+pn)*nn;//(bn*nc + pn)*nn*3;
    anchor_neighbors += (bn*nc + pn)*na*ann;
    anchor_weights += (bn*nc + pn)*na*ks*ann;
    grouped_indices += (bn*nc + pn)*nn;

     // anchor
    int cnt = 1;

    // force the sampled point into the anchor
    anchor_neighbors[a*ann] = sample_idx[bn*nc + pn];

    scalar_t kx = kernel_points[3*k];
    scalar_t ky = kernel_points[3*k + 1];
    scalar_t kz = kernel_points[3*k + 2];

    // scalar_t ww = pow(kw/radius, 2);
    // scalar_t hw = pow(2*kh/aperature, 2);
    anchor_weights[(a*ks + k)*ann] = pow(kx, 2) + pow(ky, 2) + pow(kz, 2);


    for (int nbr = 0; nbr < nn && cnt < ann; nbr++){
        scalar_t x = grouped_xyz[0*nc*nn+nbr];
        scalar_t y = grouped_xyz[1*nc*nn+nbr];
        scalar_t z = grouped_xyz[2*nc*nn+nbr];

        scalar_t norm = sqrt(x*x + y*y + z*z) + 1e-6;

        scalar_t theta = acos(x * anchors[3*a+0] / norm +
                              y * anchors[3*a+1] / norm +
                              z * anchors[3*a+2] / norm);

        // membership
        if (theta <= 0.5 * aperature) {
            // add neighbors (indexed by order of original point cloud)
            anchor_neighbors[a*ann + cnt] = grouped_indices[nbr];
            // influece weight at each kernel point
            anchor_weights[(a*ks + k)*ann + cnt] = pow(kx - x, 2) + pow(ky - y, 2) + pow(kz - z, 2);
            ++cnt;
        }

    }

    for (; cnt < ann; cnt++){
        // attach shadow points
        anchor_weights[(a*ks + k)*ann + cnt] = 1e6;
        anchor_neighbors[a*ann + cnt] = nq;
    }

}

template <typename scalar_t>
__device__ void __update(scalar_t *__restrict__ dists, int *__restrict__ dists_i,
             int idx1, int idx2) {
    const scalar_t v1 = dists[idx1], v2 = dists[idx2];
    const int i1 = dists_i[idx1], i2 = dists_i[idx2];
    dists[idx1] = max(v1, v2);
    dists_i[idx1] = v2 > v1 ? i2 : i1;
}

// furthest point sampling
// Input dataset: (b, 3, n), tmp: (b, n)
// Ouput idxs (b, m)
template <typename scalar_t, unsigned int block_size>
__global__ void furthest_point_sampling_cuda_kernel(
    int b, int n, int m, const scalar_t *__restrict__ dataset,
    scalar_t *__restrict__ temp, int *__restrict__ idxs) {

    if (m <= 0)
        return;
    __shared__ scalar_t dists[block_size];
    __shared__ int dists_i[block_size];

    int batch_index = blockIdx.x;
    dataset += batch_index * 3 * n;
    temp += batch_index * n;
    idxs += batch_index * m;

    int tid = threadIdx.x;
    const int stride = block_size;

    int old = 0;
    if (threadIdx.x == 0)
        idxs[0] = old;

    __syncthreads();
    for (int j = 1; j < m; j++) {
    int besti = 0;
    scalar_t best = -1;
    scalar_t x1 = dataset[0 * n + old];
    scalar_t y1 = dataset[1 * n + old];
    scalar_t z1 = dataset[2 * n + old];
    for (int k = tid; k < n; k += stride) {
        scalar_t x2, y2, z2;
        x2 = dataset[0 * n + k];
        y2 = dataset[1 * n + k];
        z2 = dataset[2 * n + k];
        scalar_t mag = (x2 * x2) + (y2 * y2) + (z2 * z2);
        if (mag <= 1e-3)
            continue;

        scalar_t d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) +
              (z2 - z1) * (z2 - z1);

        scalar_t d2 = min(d, temp[k]);
        temp[k] = d2;
        besti = d2 > best ? k : besti;
        best = d2 > best ? d2 : best;
    }
    dists[tid] = best;
    dists_i[tid] = besti;
    __syncthreads();

    if (block_size >= 1024) {
        if (tid < 512) {
        __update(dists, dists_i, tid, tid + 512);
        }
        __syncthreads();
    }
    if (block_size >= 512) {
        if (tid < 256) {
        __update(dists, dists_i, tid, tid + 256);
        }
        __syncthreads();
    }
    if (block_size >= 256) {
        if (tid < 128) {
        __update(dists, dists_i, tid, tid + 128);
        }
        __syncthreads();
    }
    if (block_size >= 128) {
        if (tid < 64) {
        __update(dists, dists_i, tid, tid + 64);
        }
        __syncthreads();
    }
    if (block_size >= 64) {
        if (tid < 32) {
        __update(dists, dists_i, tid, tid + 32);
        }
        __syncthreads();
    }
    if (block_size >= 32) {
        if (tid < 16) {
        __update(dists, dists_i, tid, tid + 16);
        }
        __syncthreads();
    }
    if (block_size >= 16) {
        if (tid < 8) {
        __update(dists, dists_i, tid, tid + 8);
        }
        __syncthreads();
    }
    if (block_size >= 8) {
        if (tid < 4) {
        __update(dists, dists_i, tid, tid + 4);
        }
        __syncthreads();
    }
    if (block_size >= 4) {
        if (tid < 2) {
        __update(dists, dists_i, tid, tid + 2);
        }
        __syncthreads();
    }
    if (block_size >= 2) {
        if (tid < 1) {
        __update(dists, dists_i, tid, tid + 1);
        }
        __syncthreads();
    }

    old = dists_i[0];
    if (tid == 0)
        idxs[j] = old;
    }
}

}


at::Tensor ball_query_cuda(int b, int n, int m, float radius,
                    int nsample,
                    at::Tensor new_xyz,
                    at::Tensor xyz,
                    at::Tensor idx) {

    AT_DISPATCH_FLOATING_TYPES(xyz.type(), "ball_query_cuda", ([&] {
      ball_query_cuda_kernel<scalar_t><<<b, opt_n_threads(m)>>>(b,n,m,radius,nsample,
          new_xyz.data<scalar_t>(),
          xyz.data<scalar_t>(),
          idx.data<int>()
    );
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
            printf("Error: %s\n", cudaGetErrorString(err));

    return idx;
}

std::vector<at::Tensor> anchor_query_cuda(
    at::Tensor sample_idx, // [b, np]
    at::Tensor grouped_indices, // [b, np, nn]
    at::Tensor grouped_xyz, // [b, 3, np, nn]
    at::Tensor anchors, // [na, 3]
    at::Tensor kernel_points, // [ks, 2]
    at::Tensor anchor_weights, // [b, np, na, ks, nn]
    int nq) {
    const auto batch_size = grouped_xyz.size(0);
    const auto np = grouped_xyz.size(2);
    const auto nn = grouped_xyz.size(3);
    const auto na = anchors.size(0);
    const auto ks = kernel_points.size(0);

    // blocks, threads
    // const int threads = 1024;
    // const dim3 blocks((np*nn + threads - 1) / threads, batch_size);

    AT_DISPATCH_FLOATING_TYPES(grouped_xyz.type(), "anchor_query_cuda", ([&] {
      anchor_query_cuda_kernel<scalar_t><<<batch_size, opt_n_threads(np*nn)>>>(
          sample_idx.data<int>(),
          grouped_indices.data<int>(),
          grouped_xyz.data<scalar_t>(),
          anchors.data<scalar_t>(),
          kernel_points.data<scalar_t>(),
          anchor_weights.data<scalar_t>(),
          np, nn, na, nq, ks);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
            printf("Error: %s\n", cudaGetErrorString(err));

    return {anchor_weights};
}

/*
__global__ void initial_anchor_query_cuda_kernel(
    const scalar_t* __restrict__ centers, //[b, nc, 3]
    const scalar_t* __restrict__ xyz, //[m, 3]
    const scalar_t* __restrict__ kernel_points, //[ks, na, 3]
    scalar_t* anchor_weights, //[b, ks, nc, na]
    int nb,
    int nc,
    int m,
    int na,
    int ks,
    float radius,
    float sigma)*/

std::vector<at::Tensor> initial_anchor_query_cuda(
    at::Tensor centers, // [b, 3, nc]
    at::Tensor xyz, // [m, 3]
    at::Tensor kernel_points, // [ks, na, 3]
    at::Tensor anchor_weights, // [b, ks, nc, na]
    at::Tensor anchor_ctn, // [b, ks, nc, na]
    float radius,
    float sigma
    ) {
    const auto batch_size = centers.size(0);
    const auto nc = centers.size(2);
    const auto na = kernel_points.size(1);
    const auto ks = kernel_points.size(0);
    const auto m = xyz.size(0);

    // blocks, threads
    const int threads = 1024;
    const dim3 blocks(( m*ks + threads - 1) / threads, batch_size);
    //const dim3 blocks((nc * na + threads - 1) / threads, batch_size);

    AT_DISPATCH_FLOATING_TYPES(xyz.type(), "initial_anchor_query_cuda", ([&] {
      initial_anchor_query_cuda_kernel<scalar_t><<<blocks, threads>>>(
          centers.data<scalar_t>(),
          xyz.data<scalar_t>(),
          kernel_points.data<scalar_t>(),
          anchor_weights.data<scalar_t>(),
          anchor_ctn.data<scalar_t>(),
          batch_size, nc, m, na, ks,
          radius, sigma);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
            printf("Error: %s\n", cudaGetErrorString(err));

    return {anchor_weights, anchor_ctn};
}

std::vector<at::Tensor> anchor_kp_query_cuda(
    at::Tensor sample_idx, // [b, nc]
    at::Tensor grouped_indices, // [b, nc, nn]
    at::Tensor grouped_xyz, // [b, 3, nc, nn]
    at::Tensor anchors, // [na, 3]
    at::Tensor kernel_points, // [ks, 3]
    at::Tensor anchor_neighbors, // [b, nc, na, ann]
    at::Tensor anchor_weights, // [b, nc, na, ks, ann]
    int ann,
    int nq,
    float radius,
    float aperature) {
    const auto batch_size = grouped_xyz.size(0);
    const auto nc = grouped_xyz.size(2);
    const auto nn = grouped_xyz.size(3);
    const auto na = anchors.size(0);
    const auto ks = kernel_points.size(0);

    // blocks, threads
    const int threads = 1024;
    const dim3 blocks((nc*na*ks + threads - 1) / threads, batch_size);
    //const dim3 blocks((nc * na + threads - 1) / threads, batch_size);

    AT_DISPATCH_FLOATING_TYPES(grouped_xyz.type(), "anchor_kp_query_cuda", ([&] {
      anchor_kp_query_cuda_kernel<scalar_t><<<blocks, threads>>>(
          sample_idx.data<int>(),
          grouped_indices.data<int>(),
          grouped_xyz.data<scalar_t>(),
          anchors.data<scalar_t>(),
          kernel_points.data<scalar_t>(),
          anchor_neighbors.data<int>(),
          anchor_weights.data<scalar_t>(),
          nc, nn, na, nq, ks, ann,
          radius, aperature);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
            printf("Error: %s\n", cudaGetErrorString(err));

    return {anchor_neighbors, anchor_weights};
}



at::Tensor furthest_point_sampling_cuda(
    at::Tensor source,      // [b, 3, nq]
    at::Tensor temp,        // [b, nq]
    at::Tensor sampled_idx, // [b, m]
    const int m){

    const int nb = source.size(0);
    const int nq = source.size(2);

    unsigned int n_threads = opt_n_threads(nq);

    switch (n_threads) {
    case 1024:
    AT_DISPATCH_FLOATING_TYPES(source.type(), "furthest_point_sampling_cuda", ([&] {
      furthest_point_sampling_cuda_kernel<scalar_t, 1024><<<nb, n_threads>>>(nb,nq,m,
          source.data<scalar_t>(),
          temp.data<scalar_t>(),
          sampled_idx.data<int>());
    }));
    break;
    case 512:
    AT_DISPATCH_FLOATING_TYPES(source.type(), "furthest_point_sampling_cuda", ([&] {
      furthest_point_sampling_cuda_kernel<scalar_t, 512><<<nb, n_threads>>>(nb,nq,m,
          source.data<scalar_t>(),
          temp.data<scalar_t>(),
          sampled_idx.data<int>());
    }));
    break;
    case 256:
    AT_DISPATCH_FLOATING_TYPES(source.type(), "furthest_point_sampling_cuda", ([&] {
      furthest_point_sampling_cuda_kernel<scalar_t, 256><<<nb, n_threads>>>(nb,nq,m,
          source.data<scalar_t>(),
          temp.data<scalar_t>(),
          sampled_idx.data<int>());
    }));
    break;
    case 128:
    AT_DISPATCH_FLOATING_TYPES(source.type(), "furthest_point_sampling_cuda", ([&] {
      furthest_point_sampling_cuda_kernel<scalar_t, 128><<<nb, n_threads>>>(nb,nq,m,
          source.data<scalar_t>(),
          temp.data<scalar_t>(),
          sampled_idx.data<int>());
    }));
    break;
    case 64:
    AT_DISPATCH_FLOATING_TYPES(source.type(), "furthest_point_sampling_cuda", ([&] {
      furthest_point_sampling_cuda_kernel<scalar_t, 64><<<nb, n_threads>>>(nb,nq,m,
          source.data<scalar_t>(),
          temp.data<scalar_t>(),
          sampled_idx.data<int>());
    }));
    break;
    case 32:
    AT_DISPATCH_FLOATING_TYPES(source.type(), "furthest_point_sampling_cuda", ([&] {
      furthest_point_sampling_cuda_kernel<scalar_t, 32><<<nb, n_threads>>>(nb,nq,m,
          source.data<scalar_t>(),
          temp.data<scalar_t>(),
          sampled_idx.data<int>());
    }));
    break;
    case 16:
    AT_DISPATCH_FLOATING_TYPES(source.type(), "furthest_point_sampling_cuda", ([&] {
      furthest_point_sampling_cuda_kernel<scalar_t, 16><<<nb, n_threads>>>(nb,nq,m,
          source.data<scalar_t>(),
          temp.data<scalar_t>(),
          sampled_idx.data<int>());
    }));
    break;
    case 8:
    AT_DISPATCH_FLOATING_TYPES(source.type(), "furthest_point_sampling_cuda", ([&] {
      furthest_point_sampling_cuda_kernel<scalar_t, 8><<<nb, n_threads>>>(nb,nq,m,
          source.data<scalar_t>(),
          temp.data<scalar_t>(),
          sampled_idx.data<int>());
    }));
    break;
    case 4:
    AT_DISPATCH_FLOATING_TYPES(source.type(), "furthest_point_sampling_cuda", ([&] {
      furthest_point_sampling_cuda_kernel<scalar_t, 4><<<nb, n_threads>>>(nb,nq,m,
          source.data<scalar_t>(),
          temp.data<scalar_t>(),
          sampled_idx.data<int>());
    }));
    break;
    case 2:
    AT_DISPATCH_FLOATING_TYPES(source.type(), "furthest_point_sampling_cuda", ([&] {
      furthest_point_sampling_cuda_kernel<scalar_t, 2><<<nb, n_threads>>>(nb,nq,m,
          source.data<scalar_t>(),
          temp.data<scalar_t>(),
          sampled_idx.data<int>());
    }));
    break;
    case 1:
    AT_DISPATCH_FLOATING_TYPES(source.type(), "furthest_point_sampling_cuda", ([&] {
      furthest_point_sampling_cuda_kernel<scalar_t, 1><<<nb, n_threads>>>(nb,nq,m,
          source.data<scalar_t>(),
          temp.data<scalar_t>(),
          sampled_idx.data<int>());
    }));
    break;
    default:
    AT_DISPATCH_FLOATING_TYPES(source.type(), "furthest_point_sampling_cuda", ([&] {
      furthest_point_sampling_cuda_kernel<scalar_t, 512><<<nb, n_threads>>>(nb,nq,m,
          source.data<scalar_t>(),
          temp.data<scalar_t>(),
          sampled_idx.data<int>());
    }));
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
            printf("Error: %s\n", cudaGetErrorString(err));

    return sampled_idx;
}