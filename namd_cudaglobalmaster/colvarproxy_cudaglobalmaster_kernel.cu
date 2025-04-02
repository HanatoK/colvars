#include "colvarproxy_cudaglobalmaster_kernel.h"
#if __CUDACC_VER_MAJOR__ >= 11
#include <cub/cub.cuh>
#else
#include <namd_cub/cub.cuh>
#endif

__global__ void transpose_to_host_rvector_kernel(
  const double* __restrict d_data_in,
  cvm::rvector* __restrict h_data_out,
  const int num_atoms) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < num_atoms) {
    h_data_out[i].x = d_data_in[i];
    h_data_out[i].y = d_data_in[i + num_atoms];
    h_data_out[i].z = d_data_in[i + num_atoms * 2];
  }
}

void transpose_to_host_rvector(
  const double* d_data_in,
  cvm::rvector* h_data_out,
  const int num_atoms,
  cudaStream_t stream) {
  const int block_size = 128;
  const int grid = (num_atoms + block_size - 1) / block_size;
  if (grid == 0) return;
  transpose_to_host_rvector_kernel<<<grid, block_size, 0, stream>>>(
    d_data_in, h_data_out, num_atoms);
}

__global__ void transpose_from_host_rvector_kernel(
  double*             __restrict d_data_out,
  const cvm::rvector* __restrict h_data_in,
  const int num_atoms) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < num_atoms) {
    d_data_out[i]                 = h_data_in[i].x;
    d_data_out[i + num_atoms]     = h_data_in[i].y;
    d_data_out[i + num_atoms * 2] = h_data_in[i].z;
  }
}

void transpose_from_host_rvector(
  double* d_data_out,
  const cvm::rvector* h_data_in,
  const int num_atoms,
  cudaStream_t stream) {
  const int block_size = 128;
  const int grid = (num_atoms + block_size - 1) / block_size;
  if (grid == 0) return;
  transpose_from_host_rvector_kernel<<<grid, block_size, 0, stream>>>(
    d_data_out, h_data_in, num_atoms);
}

__global__ void copy_float_to_host_double_kernel(
  const float*  __restrict d_data_in,
  cvm::real*    __restrict h_data_out,
  const int num_atoms) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < num_atoms) {
    h_data_out[i] = cvm::real(d_data_in[i]);
  }
}

void copy_float_to_host_double(
  const float* d_data_in,
  cvm::real* h_data_out,
  const int num_atoms,
  cudaStream_t stream) {
  const int block_size = 128;
  const int grid = (num_atoms + block_size - 1) / block_size;
  if (grid == 0) return;
  copy_float_to_host_double_kernel<<<grid, block_size, 0, stream>>>(
    d_data_in, h_data_out, num_atoms);
}

template <int block_size>
__global__ void compute_virial_extForce_kernel(
  const double* __restrict d_positions,
  const double* __restrict d_applied_forces,
  cudaTensor* __restrict h_virial,
  cudaTensor* __restrict d_virial,
  Vector* __restrict h_extForce,
  Vector* __restrict d_extForce,
  unsigned int* __restrict tbcatomic,
  const int num_atoms) {
  double3 r_netForce = {0, 0, 0};
  cudaTensor r_virial;
  r_virial.xx = 0.0; r_virial.xy = 0.0; r_virial.xz = 0.0;
  r_virial.yx = 0.0; r_virial.yy = 0.0; r_virial.yz = 0.0;
  r_virial.zx = 0.0; r_virial.zy = 0.0; r_virial.zz = 0.0;
  int totaltb = gridDim.x;
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  __shared__ bool isLastBlockDone;
  if(threadIdx.x == 0){
    isLastBlockDone = 0;
  }
  __syncthreads();
  if (i < num_atoms) {
    const double3 pos{
      d_positions[i],
      d_positions[i+num_atoms],
      d_positions[i+2*num_atoms]};
    const double3 f{
      d_applied_forces[i],
      d_applied_forces[i+num_atoms],
      d_applied_forces[i+2*num_atoms]};
    r_virial.xx = f.x * pos.x;
    r_virial.xy = f.x * pos.y;
    r_virial.xz = f.x * pos.z;
    r_virial.yx = f.y * pos.x;
    r_virial.yy = f.y * pos.y;
    r_virial.yz = f.y * pos.z;
    r_virial.zx = f.z * pos.x;
    r_virial.zy = f.z * pos.y;
    r_virial.zz = f.z * pos.z;
    r_netForce.x = f.x;
    r_netForce.y = f.y;
    r_netForce.z = f.z;
  }
  __syncthreads();

  typedef cub::BlockReduce<double, block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  r_netForce.x = BlockReduce(temp_storage).Sum(r_netForce.x);
  __syncthreads();
  r_netForce.y = BlockReduce(temp_storage).Sum(r_netForce.y);
  __syncthreads();
  r_netForce.z = BlockReduce(temp_storage).Sum(r_netForce.z);
  __syncthreads();

  r_virial.xx = BlockReduce(temp_storage).Sum(r_virial.xx);
  __syncthreads();
  r_virial.xy = BlockReduce(temp_storage).Sum(r_virial.xy);
  __syncthreads();
  r_virial.xz = BlockReduce(temp_storage).Sum(r_virial.xz);
  __syncthreads();

  r_virial.yx = BlockReduce(temp_storage).Sum(r_virial.yx);
  __syncthreads();
  r_virial.yy = BlockReduce(temp_storage).Sum(r_virial.yy);
  __syncthreads();
  r_virial.yz = BlockReduce(temp_storage).Sum(r_virial.yz);
  __syncthreads();

  r_virial.zx = BlockReduce(temp_storage).Sum(r_virial.zx);
  __syncthreads();
  r_virial.zy = BlockReduce(temp_storage).Sum(r_virial.zy);
  __syncthreads();
  r_virial.zz = BlockReduce(temp_storage).Sum(r_virial.zz);
  __syncthreads();

  if(threadIdx.x == 0){
    atomicAdd(&(d_virial->xx), r_virial.xx);
    atomicAdd(&(d_virial->xy), r_virial.xy);
    atomicAdd(&(d_virial->xz), r_virial.xz);

    atomicAdd(&(d_virial->yx), r_virial.yx);
    atomicAdd(&(d_virial->yy), r_virial.yy);
    atomicAdd(&(d_virial->yz), r_virial.yz);

    atomicAdd(&(d_virial->zx), r_virial.zx);
    atomicAdd(&(d_virial->zy), r_virial.zy);
    atomicAdd(&(d_virial->zz), r_virial.zz);

    atomicAdd(&(d_extForce->x), r_netForce.x);
    atomicAdd(&(d_extForce->y), r_netForce.y);
    atomicAdd(&(d_extForce->z), r_netForce.z);

    __threadfence();
    unsigned int value = atomicInc(&tbcatomic[0], totaltb);
    isLastBlockDone = (value == (totaltb -1));
  }
  __syncthreads();

  if(isLastBlockDone){
    if(threadIdx.x == 0){
      h_virial->xx = d_virial->xx;
      h_virial->xy = d_virial->xy;
      h_virial->xz = d_virial->xz;
      h_virial->yx = d_virial->yx;
      h_virial->yy = d_virial->yy;
      h_virial->yz = d_virial->yz;
      h_virial->zx = d_virial->zx;
      h_virial->zy = d_virial->zy;
      h_virial->zz = d_virial->zz;

      //reset the device virial value
      d_virial->xx = 0;
      d_virial->xy = 0;
      d_virial->xz = 0;
      d_virial->yx = 0;
      d_virial->yy = 0;
      d_virial->yz = 0;
      d_virial->zx = 0;
      d_virial->zy = 0;
      d_virial->zz = 0;

      h_extForce->x  = d_extForce->x;
      h_extForce->y  = d_extForce->y;
      h_extForce->z  = d_extForce->z;
      d_extForce->x =0 ;
      d_extForce->y =0 ;
      d_extForce->z =0 ;
      //resets atomic counter
      tbcatomic[0] = 0;
      __threadfence();
    }
  }
}

void compute_virial_extForce(
  const double* d_positions,
  const double* d_applied_forces,
  cudaTensor* h_virial,
  cudaTensor* d_virial,
  Vector* h_extForce,
  Vector* d_extForce,
  unsigned int* d_tbcatomic,
  const int num_atoms,
  cudaStream_t stream) {
  const int block_size = 128;
  const int grid = (num_atoms + block_size - 1) / block_size;
  if (grid == 0) return;
  compute_virial_extForce_kernel<block_size><<<grid, block_size, 0, stream>>>(
    d_positions, d_applied_forces,
    h_virial, d_virial,
    h_extForce, d_extForce,
    d_tbcatomic, num_atoms);
}
