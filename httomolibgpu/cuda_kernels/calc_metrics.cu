/*********************************************************************
 * Calculate correlation-based metrics for the find_center_360 method
 *********************************************************************
 *
 * The core of the find_center_360 method is calculating correlation coefficients
 * between 2 or 3 shifted matrices, over many shifting positions.
 *
 * This file has cuda kernels for this purpose, which provide speedups of > 300x
 * compared to using a straight numpy to cupy port.
 *
 * The key is the formula to calculate the Peason correlation coefficient.
 * This is calculated manually for every shifted matrix position in the same kernel.
 *
 * The correlation coefficient between two vectors (we flatten the matrices) is:
 *
 * m1_norm = m1 - mean(m1)
 * m2_norm = m2 - mean(m2)
 * m1_sqr = dot(m1_norm, m1_norm)
 * m2_sqr = dot(m2_norm, m2_norm)
 * m1_m2  = dot(m1_norm, m2_norm)
 * r = m1_m2 / sqrt(m1_sqr * m2_sqr)
 *
 * The kernels in the following compute these directly pretty much, taking into
 * consideration normalisation, overlaps, and position offsets. Also note that the
 * version with overlap requries 3 correlation coefficients (between 3 matrices).
 */


/** Function to perform a binary sum reduction for N-dimensional array storage.
 * Note that the shared_mem pointer must have space for N * BLOCK_DIM elements.
*/
template <int N, int BLOCK_DIM=128>
__device__ inline
void sum_reduction_n(float* shared_mem, float v[N]) {
    int tid = threadIdx.x;

    float *smem[N];
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        smem[i] = shared_mem + i * BLOCK_DIM;
        smem[i][tid] = v[i];
    }

    __syncthreads();
    int nt = BLOCK_DIM;
    int c = nt;
    while (c > 1)
    {
        int half = c / 2;
        if (tid < half)
        {
            #pragma unroll
            for (int i = 0; i < N; ++i) {
                smem[i][tid] += smem[i][c - tid - 1];
            }
        }
        __syncthreads();
        c = c - half;
    }

    // write back
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        v[i] = smem[i][0];
    }
}

inline __device__
float clip(float x, float min, float max) {
    x = x < -1.0f ? -1.0f : x;
    x = x > 1.0f ? 1.0f : x;
    return x;
}

__device__ inline
float sum_abs_row(const float* row, int win_width)
{
    float sum_abs = 0.0;
    for (int x = 0; x < win_width; ++x) {
        sum_abs += abs(row[x]);
    }
    return sum_abs;
}


/** Compute function without overlap, where correlation metrics with 2 matrices are calculated. */
template <bool norm>
__device__ void _calc_metrics_no_overlap(const float *mat1, int mat1_nx,
                                         const float *mat2, int mat2_nx,
                                         int win_width, int rows, int side,
                                         float *list_metric)
{
    // rows of the matrix
    const int tid = threadIdx.x;
    // position in list_pos
    const int i = blockIdx.y;
    const int npos = gridDim.y;

    const int pos = win_width / 2 + i;

    // offset matrices for position
    const float* mat2_roi = side == 1 ? mat2 : mat2 + mat2_nx - win_width;
    const float* mat1_roi = mat1 + (pos - win_width / 2);

    extern __shared__ float smem[];

    // we store our data for reductions here
    float v[3];

    ////////////////////////
    // 1. We  need the mean of the 2 matrices (flattend)
    v[0] = 0.0f;
    v[1] = 0.0f;

    for (int y = tid; y < rows; y += blockDim.x)
    {
        float norm_factor = 1.0f;
        if (norm) {
            norm_factor = sum_abs_row(&mat2_roi[y * mat2_nx], win_width) /
                sum_abs_row(&mat1_roi[y * mat1_nx], win_width);
        }
        for (int x = 0; x < win_width; ++x)
        {
            v[0] += mat1_roi[y * mat1_nx + x] * norm_factor;
            v[1] += mat2_roi[y * mat2_nx + x];
        }
    }

    // now reduce them to calc the mean
    sum_reduction_n<2>(smem, v);

    float mean_mat1 = v[0] / rows / win_width;
    float mean_mat2 = v[1] / rows / win_width;

    ///////////////////////////////////
    // 2. Calculate the sum of the dot and cross-products for the 3 matrices:
    v[0] = 0.0f;    // dot(mat1, mat1)
    v[1] = 0.0f;    // dot(mat2, mat2)
    v[2] = 0.0f;    // dot(mat1, mat2)

    for (int y = tid; y < rows; y += blockDim.x)
    {
        float norm_factor = 1.0f;
        if (norm) {
            norm_factor = sum_abs_row(&mat2_roi[y * mat2_nx], win_width) /
                sum_abs_row(&mat1_roi[y * mat1_nx], win_width);
        }
        for (int x = 0; x < win_width; ++x)
        {
            float mat1_roi_val = mat1_roi[y * mat1_nx + x] * norm_factor;
            float mat2_roi_val = mat2_roi[y * mat2_nx + x];
            mat1_roi_val -= mean_mat1;
            mat2_roi_val -= mean_mat2;
            v[0] += mat1_roi_val * mat1_roi_val;
            v[1] += mat2_roi_val * mat2_roi_val;
            v[2] += mat1_roi_val * mat2_roi_val;
        }
    }

    // now reduce them to calc the mean
    sum_reduction_n<3>(smem, v);

    ////////////////////////////////
    // 3. Calculate the correlation coefficients from the covariance values
    if (tid == 0)
    {
        // we actually need the mean of the squares
        float mat1_mat1 = v[0] / (rows * win_width - 1);
        float mat2_mat2 = v[1] / (rows * win_width - 1);
        float mat1_mat2 = v[2] / (rows * win_width - 1);
        // not calculate correlation coeffiecient
        float r = mat1_mat2 / sqrt(mat1_mat1 * mat2_mat2);
        r = clip(r, -1.0f, 1.0f);
        // metric
        float metric = abs(1.0f - r);
        list_metric[i] = metric;
    }
}


/** Compute function with overlap, where correlation metrics with 3 matrices are calculated. */
template <bool norm>
__device__ void _calc_metrics_overlap(const float *mat1, int mat1_nx,
                                      const float *mat2, int mat2_nx,
                                      int win_width, int rows, int side,
                                      float *list_metric)
{
    // rows of the matrix
    const int tid = threadIdx.x;
    // position in list_pos
    const int i = blockIdx.y;
    const int npos = gridDim.y;

    const int pos = win_width / 2 + i;

    // offset matrices for position
    const float* mat2_roi = side == 1 ? mat2 : mat2 + mat2_nx - win_width;
    const float* mat1_roi = mat1 + (pos - win_width / 2);

    extern __shared__ float smem[];

    // we need to space for 6 sum reductions for calculating the correlation coefficient
    float v[6];

    float d_ramp = 1.0f / (win_width - 1);

    ////////////////////////
    // 1. We  need the mean of the 3 matrices (flattend)
    v[0] = 0.0f;
    v[1] = 0.0f;
    v[2] = 0.0f;
    for (int y = tid; y < rows; y += blockDim.x)
    {
        float norm_factor = 1.0f;
        if (norm) {
            norm_factor = sum_abs_row(&mat2_roi[y * mat2_nx], win_width) /
                sum_abs_row(&mat1_roi[y * mat1_nx], win_width);
        }
        for (int x = 0; x < win_width; ++x)
        {
            float ramp_down = 1.0f - (x * d_ramp);
            float ramp_up = 1.0f - ramp_down;
            float mat1_roi_val = mat1_roi[y * mat1_nx + x] * norm_factor;
            float mat2_roi_val = mat2_roi[y * mat2_nx + x];
            float mat_comb_val = side == 1 ?
                (mat1_roi_val * ramp_down + mat2_roi_val * ramp_up) :
                (mat1_roi_val * ramp_up + mat2_roi_val * ramp_down);

            v[0] += mat1_roi_val;
            v[1] += mat2_roi_val;
            v[2] += mat_comb_val;
        }
    }

    sum_reduction_n<3>(smem, v);

    float mean_mat1 = v[0] / rows / win_width;
    float mean_mat2 = v[1] / rows / win_width;
    float mean_mat3 = v[2] / rows / win_width;

    ///////////////////////////////////
    // 2. Calculate the sum of the dot and cross-products for the 3 matrices:
    v[0] = 0.0f;    // dot(mat1, mat1)
    v[1] = 0.0f;    // dot(mat2, mat2)
    v[2] = 0.0f;    // dot(mat_comb, mat_comb)
    v[3] = 0.0f;    // dot(mat1, mat2)
    v[4] = 0.0f;    // dot(mat1, mat_comb)
    v[5] = 0.0f;    // dot(mat2, mat_comb)

    for (int y = tid; y < rows; y += blockDim.x)
    {
        float norm_factor = 1.0f;
        if (norm) {
            norm_factor = sum_abs_row(&mat2_roi[y * mat2_nx], win_width) /
                sum_abs_row(&mat1_roi[y * mat1_nx], win_width);
        }
        for (int x = 0; x < win_width; ++x)
        {
            float ramp_down = 1.0f - (x * d_ramp);
            float ramp_up = 1.0f - ramp_down;
            float mat1_roi_val = mat1_roi[y * mat1_nx + x] * norm_factor;
            float mat2_roi_val = mat2_roi[y * mat2_nx + x];
            float mat_comb_val = side == 1 ?
                (mat1_roi_val * ramp_down + mat2_roi_val * ramp_up) :
                (mat1_roi_val * ramp_up + mat2_roi_val * ramp_down);

            // for covariance matrix, we need to remove the mean first
            mat1_roi_val -= mean_mat1;
            mat2_roi_val -= mean_mat2;
            mat_comb_val -= mean_mat3;

            // now sum the products
            v[0] += mat1_roi_val * mat1_roi_val;
            v[1] += mat2_roi_val * mat2_roi_val;
            v[2] += mat_comb_val * mat_comb_val;
            v[3] += mat1_roi_val * mat2_roi_val;
            v[4] += mat1_roi_val * mat_comb_val;
            v[5] += mat2_roi_val * mat_comb_val;
        }

    }

    // 6 smem reductions
    sum_reduction_n<6>(smem, v);

    ////////////////////////////////
    // 3. Calculate the correlation coefficients from the covariance values
    if (tid == 0)
    {
        // mean values
        float mat1_mat1 = v[0] / (rows * win_width - 1);
        float mat2_mat2 = v[1] / (rows * win_width - 1);
        float mat3_mat3 = v[2] / (rows * win_width - 1);
        float mat1_mat2 = v[3] / (rows * win_width - 1);
        float mat1_mat3 = v[4] / (rows * win_width - 1);
        float mat2_mat3 = v[5] / (rows * win_width - 1);
        // normalise to get correlation coefficients
        float r12 = mat1_mat2 / sqrt(mat1_mat1 * mat2_mat2);
        float r13 = mat1_mat3 / sqrt(mat1_mat1 * mat3_mat3);
        float r23 = mat2_mat3 / sqrt(mat2_mat2 * mat3_mat3);
        // clip
        r12 = clip(r12, -1.0f, 1.0f);
        r13 = clip(r13, -1.0f, 1.0f);
        r23 = clip(r23, -1.0f, 1.0f);
        // metric
        float metric_1 = abs(1.0f - r12);
        float metric_2 = abs(1.0f - r23);
        float metric_3 = abs(1.0f - r13);
        // average and output
        list_metric[i] = (metric_1 + metric_2 + metric_3) / 3.0f;
    }
}





/** Main entry point - it calls one of the two variants above.
 *
 * We use a template here, so that one of the two branches gets completely eliminated by
 * the compiler (rather than at runtime), which reduces the register count.
 */
template <bool norm, bool use_overlap>
__global__ void calc_metrics_kernel(const float *mat1, int mat1_nx,
                                    const float *mat2, int mat2_nx,
                                    int win_width, int rows, int side,
                                    float *list_metric)
{
    if (use_overlap) {
        _calc_metrics_overlap<norm>(mat1, mat1_nx, mat2, mat2_nx, win_width, rows, side, list_metric);
    } else {
        _calc_metrics_no_overlap<norm>(mat1, mat1_nx, mat2, mat2_nx, win_width, rows, side, list_metric);
    }
}