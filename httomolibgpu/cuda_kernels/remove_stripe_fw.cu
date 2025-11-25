template<int WSize>
__global__ void grouped_convolution_x(
    int dim_x,
    int dim_y,
    int dim_z,
    const float* in,
    int in_stride_x,
    int in_stride_y,
    int in_stride_z,
    float* out,
    int out_stride_z,
    int out_stride_group,
    const float* w
)
{
    const int g_thd_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int g_thd_y = blockDim.y * blockIdx.y + threadIdx.y;
    const int g_thd_z = blockDim.z * blockIdx.z + threadIdx.z;
    if (g_thd_x >= dim_x || g_thd_y >= dim_y || g_thd_z >= dim_z)
    {
        return;
    }

    constexpr int out_groups = 2;
    for (int i = 0; i < out_groups; ++i)
    {
        float acc = 0.F;
        for (int j = 0; j < WSize; ++j)
        {
            const int w_idx = i * WSize + j;
            const int in_idx = (g_thd_x * in_stride_x + j) + g_thd_y * in_stride_y + g_thd_z * in_stride_z;
            acc += w[w_idx] * in[in_idx];
        }
        const int out_idx = g_thd_x + g_thd_y * dim_x + g_thd_z * out_stride_z + i * out_stride_group;
        out[out_idx] = acc;
    }
}

template<int WSize>
__global__ void grouped_convolution_y(
    int dim_x,
    int dim_y,
    int dim_z,
    const float* in,
    int in_stride_x,
    int in_stride_y,
    int in_stride_z,
    int in_stride_group,
    float* out,
    int out_stride_z,
    int out_stride_group,
    const float* w
)
{
    const int g_thd_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int g_thd_y = blockDim.y * blockIdx.y + threadIdx.y;
    const int g_thd_z = blockDim.z * blockIdx.z + threadIdx.z;
    if (g_thd_x >= dim_x || g_thd_y >= dim_y || g_thd_z >= dim_z)
    {
        return;
    }

    constexpr int in_groups = 2;
    constexpr int out_groups = 2;
    constexpr int item_stride_y = 2;
    for (int group = 0; group < in_groups; ++group)
    {
        for (int i = 0; i < out_groups; ++i)
        {
            float acc = 0.F;
            for (int j = 0; j < WSize; ++j)
            {
                const int w_idx = (out_groups * group + i) * WSize + j;
                const int in_idx = g_thd_x * in_stride_x + (item_stride_y * g_thd_y + j) * in_stride_y + group * in_stride_group + g_thd_z * in_stride_z;
                acc += w[w_idx] * in[in_idx];
            }
            const int out_idx = g_thd_x + g_thd_y * dim_x + g_thd_z * out_stride_z + (out_groups * group + i) * out_stride_group;
            out[out_idx] = acc;
        }
    }
}

template<int WSize>
__global__ void transposed_convolution_x(
    int dim_x,
    int dim_y,
    int dim_z,
    const float* in,
    int in_dim_x,
    int in_stride_y,
    int in_stride_z,
    const float* w,
    float* out
)
{
    const int g_thd_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int g_thd_y = blockDim.y * blockIdx.y + threadIdx.y;
    const int g_thd_z = blockDim.z * blockIdx.z + threadIdx.z;
    if (g_thd_x >= dim_x || g_thd_y >= dim_y || g_thd_z >= dim_z)
    {
        return;
    }

    constexpr int item_out_stride = 2;
    float acc = 0.F;
    for (int i = 0; i < WSize; ++i)
    {
        const int in_x = (g_thd_x - i) / item_out_stride;
        const int in_x_mod = (g_thd_x - i) % item_out_stride;
        if (in_x_mod == 0 && in_x >= 0 && in_x < in_dim_x)
        {
            const int in_idx = in_x + g_thd_y * in_stride_y + g_thd_z * in_stride_z;
            acc += in[in_idx] * w[i];
        }
    }
    const int out_idx = g_thd_x + dim_x * g_thd_y + dim_x * dim_y * g_thd_z;
    out[out_idx] = acc;
}

template<int WSize>
__global__ void transposed_convolution_y(
    int dim_x,
    int dim_y,
    int dim_z,
    const float* in,
    int in_dim_y,
    int in_stride_y,
    int in_stride_z,
    const float* w,
    float* out
)
{
    const int g_thd_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int g_thd_y = blockDim.y * blockIdx.y + threadIdx.y;
    const int g_thd_z = blockDim.z * blockIdx.z + threadIdx.z;
    if (g_thd_x >= dim_x || g_thd_y >= dim_y || g_thd_z >= dim_z)
    {
        return;
    }

    constexpr int item_out_stride = 2;
    float acc = 0.F;
    for (int i = 0; i < WSize; ++i)
    {
        const int in_y = (g_thd_y - i) / item_out_stride;
        const int in_y_mod = (g_thd_y - i) % item_out_stride;
        if (in_y_mod == 0 && in_y >= 0 && in_y < in_dim_y)
        {
            const int in_idx = g_thd_x + in_y * in_stride_y + g_thd_z * in_stride_z;
            acc += in[in_idx] * w[i];
        }
    }
    const int out_idx = g_thd_x + dim_x * g_thd_y + dim_x * dim_y * g_thd_z;
    out[out_idx] = acc;
}
