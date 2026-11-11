#include <cupy/complex.cuh>

template <typename Type>
__device__ __forceinline__ void
raven_filter_impl(
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
  if( abs(pz - centerz) <= v0 ) {

    // fftshift places zero frequency at width / 2 for both odd and even widths.
    // Raven's transfer function is real: H = 1 / (1 + (u/u0)^(2*n)).
    // See also Farzi et al., IEEE TMI 40 (2021), Eq. (14),
    // doi:10.1109/TMI.2021.3097907. No phase rotation is applied.
    Type base = Type(abs(px - centerx)) / Type(u0);
    // Reciprocal form outside the passband avoids overflowing large powers.
    Type power = pow(base > Type(1) ? Type(1) / base : base,
                     Type(2) * Type(n));
    Type filtered_value = base > Type(1)
        ? power / (Type(1) + power) : Type(1) / (Type(1) + power);
    value *= filtered_value;
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

extern "C" __global__ void raven_filter_float(complex<float> *input,
                                              complex<float> *output,
                                              int width, int images, int height,
                                              int u0, int n, int v0)
{
  raven_filter_impl<float>(input, output, width, images, height, u0, n, v0);
}

extern "C" __global__ void raven_filter_double(complex<double> *input,
                                               complex<double> *output,
                                               int width, int images, int height,
                                               int u0, int n, int v0)
{
  raven_filter_impl<double>(input, output, width, images, height, u0, n, v0);
}
