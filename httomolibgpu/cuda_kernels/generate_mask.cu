extern "C" __global__ void generate_mask(const int ncol, const int nrow,
                                         const int cen_col, const int cen_row,
                                         const float du, const float dv,
                                         const float radius, const float drop,
                                         unsigned short *mask) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockIdx.y;

  if (i >= ncol/2+1)
    return;

  // we only need to look at the right half as we're using a real2complex FFT
  int outi = i;
  i += ncol/2-1;

  int pos = __float2int_ru(((j - cen_row) * dv / radius) / du);
  int pos1 = -pos + cen_col;
  int pos2 = pos + cen_col;

  if (pos1 > pos2) {
    int temp = pos1;
    pos1 = pos2;
    pos2 = temp;
    if (pos1 >= ncol) {
      pos1 = ncol - 1;
    }
    if (pos2 < 0) {
      pos2 = 0;
    }
  } else {
    if (pos1 < 0) {
      pos1 = 0;
    }
    if (pos2 >= ncol) {
      pos2 = ncol - 1;
    }
  }

  short outval = (pos1 <= i && i <= pos2) ? 1 : 0;

  // mask[cen_row - drop: cen_row + drop + 1, :] = 0
  if (j >= cen_row - drop && j <= cen_row + drop) {
    outval = 0;
  }
  // mask[:, cen_col - 1: cen_col + 2] = 0
  if (i >= cen_col - 1 && i <= cen_col + 1) {
    outval = 0;
  }

  mask[j * (ncol/2+1) + outi] = outval;
}