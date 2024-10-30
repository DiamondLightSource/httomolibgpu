
template <typename Type, int diameter>
__global__ void raven__filter_kernel3d(Type *in, int Z, int M, int N) {

  const long i = blockDim.x * blockIdx.x + threadIdx.x;
  const long j = blockDim.y * blockIdx.y + threadIdx.y;
  const long k = blockDim.z * blockIdx.z + threadIdx.z;

  if (i >= N || j >= M || k >= Z)
    return;

  
}
