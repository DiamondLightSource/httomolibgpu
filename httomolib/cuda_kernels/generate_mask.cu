extern "C" __global__ void generate_mask(const int ncol, const int nrow,
                                         const int cen_col, const int cen_row,
                                         const float du, const float dv,
                                         const float radius, const float drop,
                                         unsigned short *mask) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  if (j >= nrow || i >= ncol)
    return;

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

  // write to ifft-shifted positions
  int ishift = (ncol + 1) / 2;
  int jshift = (nrow + 1) / 2;
  int outi = (i + ishift) % ncol;
  int outj = (j + jshift) % nrow;

  mask[outj * ncol + outi] = outval;
}