#include <cupy/complex.cuh>

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795f
#endif

extern "C" __global__ void
paganin_filter_gen(int width1, int height1, float resolution, float wavelength,
                   float distance, float ratio, complex<float> *filtercomplex) {
  int px = threadIdx.x + blockIdx.x * blockDim.x;
  int py = threadIdx.y + blockIdx.y * blockDim.y;
  if (px >= width1)
    return;
  if (py >= height1)
    return;

  float dpx = 1.0f / (width1 * resolution);
  float dpy = 1.0f / (height1 * resolution);
  int centerx = (width1 + 1) / 2 - 1;
  int centery = (height1 + 1) / 2 - 1;

  float pxx = (px - centerx) * dpx;
  float pyy = (py - centery) * dpy;
  float pd = (pxx * pxx + pyy * pyy) * wavelength * distance * M_PI;
             ;
  float filter1 = 1.0f + ratio * pd;

  complex<float> value = 1.0f / complex<float>(filter1, filter1);

  // ifftshifting positions
  int xshift = (width1 + 1) / 2;
  int yshift = (height1 + 1) / 2;
  int outX = (px + xshift) % width1;
  int outY = (py + yshift) % height1;

  filtercomplex[outY * width1 + outX] = value;
}