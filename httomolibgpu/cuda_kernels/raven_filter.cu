#include <cupy/complex.cuh>

extern "C" __global__ void 
raven_filter(complex<float> *input, complex<float> *output, int width1, int height1, int u0, int n, int v0) {

  int centerx = (width1 + 1) / 2 - 1;
  int centery = (height1 + 1) / 2 - 1;

  int px = threadIdx.x + blockIdx.x * blockDim.x;
  int py = threadIdx.y + blockIdx.y * blockDim.y;

  if (px >= width1)
    return;
  if (py >= height1)
    return;

  complex<float> value = input[py * width1 + px];
  if( py >= (centery - v0) && py <= (centery + v0 + 1) ) {
    
    double base = (px - centerx) / u0;
    double filtered_value = base;
    for( int i = 1; i < 2 * n; i++ )
      filtered_value *= base;

    filtered_value = 1.0f / (1.0 + filtered_value);
    value *= complex<float>(filtered_value, filtered_value);
  }

  // ifftshifting positions
  int xshift = (width1 + 1) / 2;
  int yshift = (height1 + 1) / 2;
  int outX = (px + xshift) % width1;
  int outY = (py + yshift) % height1;

  output[outY * width1 + outX] = value;
}
