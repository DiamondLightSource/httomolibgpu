#include <cupy/complex.cuh>

extern "C" __global__ void
shift_whole_shifts(const float *sino2, const float *sino3,
                   const float *__restrict__ list_shift, float *mat, int nx,
                   int nymat) {
  int xid = threadIdx.x + blockIdx.x * blockDim.x;
  int yid = blockIdx.y;
  int zid = blockIdx.z;
  int ny = gridDim.y;

  if (xid >= nx)
    return;

  float shift_col = list_shift[zid];
  float int_part = 0.0;
  float frac_part = modf(shift_col, &int_part);
  if (abs(frac_part) > 1e-5f) {
    // we have a floating point shift, so we only roll in
    // sino3, but we leave the rest for later using scipy
    int shift_int =
        shift_col >= 0.0 ? int(ceil(shift_col)) : int(floor(shift_col));
    if (shift_int >= 0 && xid < shift_int) {
      mat[zid * nymat * nx + yid * nx + xid] = sino3[yid * nx + xid];
    }
    if (shift_int < 0 && xid >= nx + shift_int) {
      mat[zid * nymat * nx + yid * nx + xid] = sino3[yid * nx + xid];
    }
  } else {
    // we have an integer shift, so we can roll in directly
    // by indexing
    int shift_int = int(shift_col);
    if (shift_int >= 0) {
      if (xid >= shift_int) {
        mat[zid * nymat * nx + yid * nx + xid] =
            sino2[yid * nx + xid - shift_int];
      } else {
        mat[zid * nymat * nx + yid * nx + xid] = sino3[yid * nx + xid];
      }
    } else {
      if (xid < nx + shift_int) {
        mat[zid * nymat * nx + yid * nx + xid] =
            sino2[yid * nx + xid - shift_int];
      } else {
        mat[zid * nymat * nx + yid * nx + xid] = sino3[yid * nx + xid];
      }
    }
  }
}