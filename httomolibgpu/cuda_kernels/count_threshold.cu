template <typename Type>
__device__ __forceinline__
void count_values_threshold_kernel(
    Type *data,
    int Z,
    int M,
    int N,
    float upper_bound,
    float lower_bound,
    unsigned long long *indices_upper_bound,
    unsigned long long *indices_lower_bound)
{
    const long i = blockDim.x * blockIdx.x + threadIdx.x;
    const long j = blockDim.y * blockIdx.y + threadIdx.y;
    const long k = blockDim.z * blockIdx.z + threadIdx.z;

    if (i >= N || j >= M || k >= Z)
        return;

    const long long index =
        static_cast<long long>(i)
        + N * static_cast<long long>(j)
        + N * M * static_cast<long long>(k);

    if (data[index] > upper_bound) {
        atomicAdd(&indices_upper_bound[0], 1ULL);
    }

    if (data[index] < lower_bound) {
        atomicAdd(&indices_lower_bound[0], 1ULL);
    }
}

extern "C" __global__
void count_values_threshold(
    float *data,
    int Z,
    int M,
    int N,
    float upper_bound,
    float lower_bound,
    unsigned long long *indices_upper_bound,
    unsigned long long *indices_lower_bound)
{
    count_values_threshold_kernel<float>(
        data,
        Z,
        M,
        N,
        upper_bound,
        lower_bound,
        indices_upper_bound,
        indices_lower_bound
    );
}