template <typename Type, int diameter>
__device__ __forceinline__ void median_general_kernel3d_impl(const Type *in, Type *out, float dif,
                                                             int Z, int M, int N) {
  constexpr int radius = diameter / 2;
  constexpr int d3 = diameter * diameter * diameter;
  constexpr int midpoint = d3 / 2;

  Type ValVec[d3];
  const long i = blockDim.x * blockIdx.x + threadIdx.x;
  const long j = blockDim.y * blockIdx.y + threadIdx.y;
  const long k = blockDim.z * blockIdx.z + threadIdx.z;

  if (i >= N || j >= M || k >= Z)
    return;

  long long index = static_cast<long long>(i) + N * static_cast<long long>(j) + N * M * static_cast<long long>(k);

  int counter = 0;
  for (int i_m = -radius; i_m <= radius; i_m++) {
    long long i1 = i + i_m;   // using long long to avoid integer overflows
    if ((i1 < 0) || (i1 >= N))
      i1 = i;
    for (int j_m = -radius; j_m <= radius; j_m++) {
      long long j1 = j + j_m;
      if ((j1 < 0) || (j1 >= M))
        j1 = j;
      for (int k_m = -radius; k_m <= radius; k_m++) {
        long long k1 = k + k_m;
        if ((k1 < 0) || (k1 >= Z))
          k1 = k;
        ValVec[counter] = in[i1 + N * j1 + N * M * k1];
        counter++;
      }
    }
  }  

  /* do bubble sort here */
  for (int x = 0; x < d3 - 1; x++) {
    for (int y = 0; y < d3 - x - 1; y++) {
      if (ValVec[y] > ValVec[y + 1]) {
        Type temp = ValVec[y];
        ValVec[y] = ValVec[y + 1];
        ValVec[y + 1] = temp;
      }
    }
  }

  if (dif > 0.0f) {
    /* perform dezingering */
    out[index] =
        fabsf(in[index] - ValVec[midpoint]) >= dif ? ValVec[midpoint] : in[index];
  }
  else out[index] = ValVec[midpoint]; /* median filtering */
}

extern "C" __global__ void median_general_kernel3d_float_3(const float *in, float *out, float dif, int Z, int M, int N)
{
  median_general_kernel3d_impl<float, 3>(in, out, dif, Z, M, N);
}

extern "C" __global__ void median_general_kernel3d_float_5(const float *in, float *out, float dif, int Z, int M, int N)
{
  median_general_kernel3d_impl<float, 5>(in, out, dif, Z, M, N);
}

extern "C" __global__ void median_general_kernel3d_float_7(const float *in, float *out, float dif, int Z, int M, int N)
{
  median_general_kernel3d_impl<float, 7>(in, out, dif, Z, M, N);
}

extern "C" __global__ void median_general_kernel3d_float_9(const float *in, float *out, float dif, int Z, int M, int N)
{
  median_general_kernel3d_impl<float, 9>(in, out, dif, Z, M, N);
}

extern "C" __global__ void median_general_kernel3d_float_11(const float *in, float *out, float dif, int Z, int M, int N)
{
  median_general_kernel3d_impl<float, 11>(in, out, dif, Z, M, N);
}

extern "C" __global__ void median_general_kernel3d_float_13(const float *in, float *out, float dif, int Z, int M, int N)
{
  median_general_kernel3d_impl<float, 13>(in, out, dif, Z, M, N);
}

extern "C" __global__ void median_general_kernel3d_unsigned_short_3(const unsigned short *in, unsigned short *out, float dif, int Z, int M, int N)
{
  median_general_kernel3d_impl<unsigned short, 3>(in, out, dif, Z, M, N);
}

extern "C" __global__ void median_general_kernel3d_unsigned_short_5(const unsigned short *in, unsigned short *out, float dif, int Z, int M, int N)
{
  median_general_kernel3d_impl<unsigned short, 5>(in, out, dif, Z, M, N);
}

extern "C" __global__ void median_general_kernel3d_unsigned_short_7(const unsigned short *in, unsigned short *out, float dif, int Z, int M, int N)
{
  median_general_kernel3d_impl<unsigned short, 7>(in, out, dif, Z, M, N);
}

extern "C" __global__ void median_general_kernel3d_unsigned_short_9(const unsigned short *in, unsigned short *out, float dif, int Z, int M, int N)
{
  median_general_kernel3d_impl<unsigned short, 9>(in, out, dif, Z, M, N);
}

extern "C" __global__ void median_general_kernel3d_unsigned_short_11(const unsigned short *in, unsigned short *out, float dif, int Z, int M, int N)
{
  median_general_kernel3d_impl<unsigned short, 11>(in, out, dif, Z, M, N);
}

extern "C" __global__ void median_general_kernel3d_unsigned_short_13(const unsigned short *in, unsigned short *out, float dif, int Z, int M, int N)
{
  median_general_kernel3d_impl<unsigned short, 13>(in, out, dif, Z, M, N);
}
