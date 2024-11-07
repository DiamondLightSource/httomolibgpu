#include <cupy/complex.cuh>

extern "C" __global__ void 
raven_filter(
  complex<float> *input,
  complex<float> *output,
  int width, int images, int height, 
  int u0, int n, int v0) {

  const int px = threadIdx.x + blockIdx.x * blockDim.x;
  const int py = threadIdx.y + blockIdx.y * blockDim.y;
  const int pz = threadIdx.z + blockIdx.z * blockDim.z;

  if (px >= width || py >= images || pz >= height)
    return;

  int centerx = width / 2;
  int centerz = height / 2;

  complex<float> value = input[pz * width * images + py * width + px];
  if( pz >= (centerz - v0) && pz < (centerz + v0 + 1) ) {
    
    // +1 needed to match with CPU implementation
    float base = float(px - centerx + 1) / u0;
    float power = base;
    for( int i = 1; i < 2 * n; i++ )
      power *= base;

    float filtered_value = 1.f / (1.f + power);
    value *= complex<float>(filtered_value, filtered_value);
  }

  // ifftshifting positions
  int xshift = (width + 1) / 2;
  int zshift = (height + 1) / 2;
  int outX = (px + xshift) % width;
  int outZ = (pz + zshift) % height;

  output[outZ * width * images + py * width + outX] = value;
}
