#include <cupy/complex.cuh>

extern "C" __global__ void 
raven_filter(complex<float> *input, complex<float> *output, int width1, int height1, int u0, int n, int v0) {

  int centerx = width1 / 2;
  int centery = height1 / 2;

  int px = threadIdx.x + blockIdx.x * blockDim.x;
  int py = threadIdx.y + blockIdx.y * blockDim.y;

  if (px >= width1)
    return;
  if (py >= height1)
    return;

  complex<float> value = input[py * width1 + px];
  if( py >= (centery - v0) && py < (centery + v0 + 1) ) {
    
    // +1 needed to match with CPU implementation
    float base = float(px - centerx + 1) / u0;
    float power = base;
    for( int i = 1; i < 2 * n; i++ )
      power *= base;

    float filtered_value = 1.f / (1.f + power);
    value *= complex<float>(filtered_value, filtered_value);
  }

  // ifftshifting positions
  int xshift = (width1 + 1) / 2;
  int yshift = (height1 + 1) / 2;
  int outX = (px + xshift) % width1;
  int outY = (py + yshift) % height1;

  output[outY * width1 + outX] = value;
}
