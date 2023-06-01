extern "C" __global__ void downsample_sino(float *sino, int dx, int dz,
                                           int level, float *out) {
  // use shared memory to store the values used to "merge" columns of the
  // sinogram in the downsampling process
  extern __shared__ float downsampled_vals[];
  unsigned int binsize, i, j, k, orig_ind, out_ind, output_bin_no;
  i = blockDim.x * blockIdx.x + threadIdx.x;
  j = 0;
  k = blockDim.y * blockIdx.y + threadIdx.y;
  orig_ind = (k * dz) + i;
  binsize = 1 << level;
  unsigned int dz_downsampled =
      __float2uint_rd(fdividef(__uint2float_rd(dz), __uint2float_rd(binsize)));
  unsigned int i_downsampled =
      __float2uint_rd(fdividef(__uint2float_rd(i), __uint2float_rd(binsize)));
  if (orig_ind < dx * dz) {
    output_bin_no =
        __float2uint_rd(fdividef(__uint2float_rd(i), __uint2float_rd(binsize)));
    out_ind = (k * dz_downsampled) + i_downsampled;
    downsampled_vals[threadIdx.y * 8 + threadIdx.x] =
        sino[orig_ind] / __uint2float_rd(binsize);
    // synchronise threads within thread-block so that it's guaranteed
    // that all the required values have been copied into shared memeory
    // to then sum and save in the downsampled output
    __syncthreads();
    // arbitrarily use the "beginning thread" in each "lot" of pixels
    // for downsampling to then save the desired value in the
    // downsampled output array
    if (i % 4 == 0) {
      out[out_ind] = downsampled_vals[threadIdx.y * 8 + threadIdx.x] +
                     downsampled_vals[threadIdx.y * 8 + threadIdx.x + 1] +
                     downsampled_vals[threadIdx.y * 8 + threadIdx.x + 2] +
                     downsampled_vals[threadIdx.y * 8 + threadIdx.x + 3];
    }
  }
}