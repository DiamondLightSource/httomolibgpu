extern "C" __global__ void grouped_convolution_x(
    long long dim_x,
    long long dim_y,
    long long dim_z,
    const float* in,
    long long in_stride_x,
    long long in_stride_y,
    long long in_stride_z,
    float* out,
    long long out_stride_z,
    long long out_stride_group,
    const float* w,
    int wk
)
{
    const long long g_thd_x = blockDim.x * blockIdx.x + threadIdx.x;
    const long long g_thd_y = blockDim.y * blockIdx.y + threadIdx.y;
    const long long g_thd_z = blockDim.z * blockIdx.z + threadIdx.z;
    if (g_thd_x >= dim_x || g_thd_y >= dim_y || g_thd_z >= dim_z)
    {
        return;
    }

    constexpr int out_groups = 2;
    for (int i = 0; i < out_groups; ++i)
    {
        float acc = 0.F;
        for (int j = 0; j < wk; ++j)
        {
            const int w_idx = i * wk + j;
            const long long in_idx = (g_thd_x * in_stride_x + j) + g_thd_y * in_stride_y + g_thd_z * in_stride_z;
            acc += w[w_idx] * in[in_idx];
        }
        const long long out_idx = g_thd_x + g_thd_y * dim_x + g_thd_z * out_stride_z + i * out_stride_group;
        out[out_idx] = acc;
    }
}

extern "C" __global__ void grouped_convolution_y(
    long long dim_x,
    long long dim_y,
    long long dim_z,
    const float* in,
    long long in_stride_x,
    long long in_stride_y,
    long long in_stride_z,
    long long in_stride_group,
    float* out,
    long long out_stride_z,
    long long out_stride_group,
    const float* w,
    int hk
)
{
    const long long g_thd_x = blockDim.x * blockIdx.x + threadIdx.x;
    const long long g_thd_y = blockDim.y * blockIdx.y + threadIdx.y;
    const long long g_thd_z = blockDim.z * blockIdx.z + threadIdx.z;
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
            for (int j = 0; j < hk; ++j)
            {
                const long long w_idx = (out_groups * group + i) * hk + j;
                const long long in_idx = g_thd_x * in_stride_x + (item_stride_y * g_thd_y + j) * in_stride_y + group * in_stride_group + g_thd_z * in_stride_z;
                acc += w[w_idx] * in[in_idx];
            }
            const long long out_idx = g_thd_x + g_thd_y * dim_x + g_thd_z * out_stride_z + (out_groups * group + i) * out_stride_group;
            out[out_idx] = acc;
        }
    }
}

extern "C" __global__ void transposed_convolution_x(
    long long dim_x,
    long long dim_y,
    long long dim_z,
    const float* in,
    long long in_dim_x,
    long long in_stride_y,
    long long in_stride_z,
    const float* w,
    int wk,
    float* out
)
{
    const long long g_thd_x = blockDim.x * blockIdx.x + threadIdx.x;
    const long long g_thd_y = blockDim.y * blockIdx.y + threadIdx.y;
    const long long g_thd_z = blockDim.z * blockIdx.z + threadIdx.z;
    if (g_thd_x >= dim_x || g_thd_y >= dim_y || g_thd_z >= dim_z)
    {
        return;
    }

    constexpr int item_out_stride = 2;
    float acc = 0.F;
    for (int i = 0; i < wk; ++i)
    {
        const long long in_x = (g_thd_x - i) / item_out_stride;
        const long long in_x_mod = (g_thd_x - i) % item_out_stride;
        if (in_x_mod == 0 && in_x >= 0 && in_x < in_dim_x)
        {
            const long long in_idx = in_x + g_thd_y * in_stride_y + g_thd_z * in_stride_z;
            acc += in[in_idx] * w[i];
        }
    }
    const long long out_idx = g_thd_x + dim_x * g_thd_y + dim_x * dim_y * g_thd_z;
    out[out_idx] = acc;
}

extern "C" __global__ void transposed_convolution_y(
    long long dim_x,
    long long dim_y,
    long long dim_z,
    const float* in,
    long long in_dim_y,
    long long in_stride_y,
    long long in_stride_z,
    const float* w,
    int hk,
    float* out
)
{
    const long long g_thd_x = blockDim.x * blockIdx.x + threadIdx.x;
    const long long g_thd_y = blockDim.y * blockIdx.y + threadIdx.y;
    const long long g_thd_z = blockDim.z * blockIdx.z + threadIdx.z;
    if (g_thd_x >= dim_x || g_thd_y >= dim_y || g_thd_z >= dim_z)
    {
        return;
    }

    constexpr int item_out_stride = 2;
    float acc = 0.F;
    for (int i = 0; i < hk; ++i)
    {
        const long long in_y = (g_thd_y - i) / item_out_stride;
        const long long in_y_mod = (g_thd_y - i) % item_out_stride;
        if (in_y_mod == 0 && in_y >= 0 && in_y < in_dim_y)
        {
            const long long in_idx = g_thd_x + in_y * in_stride_y + g_thd_z * in_stride_z;
            acc += in[in_idx] * w[i];
        }
    }
    const long long out_idx = g_thd_x + dim_x * g_thd_y + dim_x * dim_y * g_thd_z;
    out[out_idx] = acc;
}
