template <typename Type>
__global__ void remove_nan_inf(Type *data, int Z, int M, int N, int *result) {
  const long i = blockDim.x * blockIdx.x + threadIdx.x;
  const long j = blockDim.y * blockIdx.y + threadIdx.y;
  const long k = blockDim.z * blockIdx.z + threadIdx.z;

  if (i >= N || j >= M || k >= Z)
    return;

  long long index = static_cast<long long>(i) + N * static_cast<long long>(j) + N * M * static_cast<long long>(k);

  float val = float(data[index]); /*needs a cast to float for isnan isinf functions to work*/
  Type zero = 0;
  if (isnan(val) || isinf(val)) {
    result[0] = 1;
    data[index] = zero;
  }

}