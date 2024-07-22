template <typename Type, int diameter>
__global__ void median_general_kernel3d(const Type *in, Type *out, float dif,
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
