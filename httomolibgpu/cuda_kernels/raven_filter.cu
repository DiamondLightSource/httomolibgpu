#include <cupy/complex.cuh>

template <typename Type>
__global__ void 
raven_filter(
  complex<Type> *input,
  complex<Type> *output,
  int width, int images, int height, 
  int u0, int n, int v0) {

  const int px = threadIdx.x + blockIdx.x * blockDim.x;
  const int py = threadIdx.y + blockIdx.y * blockDim.y;
  const int pz = threadIdx.z + blockIdx.z * blockDim.z;

  if (px >= width || py >= images || pz >= height)
    return;

  int centerx = width / 2;
  int centerz = height / 2;

  long long index = static_cast<long long>(px) + 
                    width * static_cast<long long>(py) + 
                    width * images * static_cast<long long>(pz);

  complex<Type> value = input[index];
  if( pz >= (centerz - v0) && pz < (centerz + v0 + 1) ) {
    
    // +1 needed to match with CPU implementation
    Type base = Type(px - centerx + 1) / u0;
    Type power = base;
    for( int i = 1; i < 2 * n; i++ )
      power *= base;

    Type filtered_value = 1.f / (1.f + power);
    value *= complex<Type>(filtered_value, filtered_value);
  }

  // ifftshifting positions
  int xshift = (width + 1) / 2;
  int zshift = (height + 1) / 2;
  int outX = (px + xshift) % width;
  int outZ = (pz + zshift) % height;

  long long outIndex = static_cast<long long>(outX) + 
                       width * static_cast<long long>(py) + 
                       width * images * static_cast<long long>(outZ);

  output[outIndex] = value;
}
