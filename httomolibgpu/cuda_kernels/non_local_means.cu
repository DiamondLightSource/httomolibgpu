/*
 * Copyright 2026 Diamond Light Source Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Patch-distance model inspired by CCPi-Regularisation-Toolkit:
 * src/Core/regularisers_GPU/PatchSelect_GPU_core.cu
 * Copyright 2017 Daniil Kazantsev
 * Copyright 2017 Srikanth Nagella, Edoardo Pasca
 * Copyright 2018 Diamond Light Source Ltd.
 * https://github.com/TomographicImaging/CCPi-Regularisation-Toolkit
 *
 * Computes NLM directly, without allocating a neighbour graph. Unlike the
 * reference, this retains zero-distance matches and indexes Gaussian weights
 * by patch offset even when the patch is truncated at an image boundary.
 */
template<int K, bool FULL_3D>
__global__ void non_local_means_kernel(
    const float* input, float* output, const float* gaussian,
    const long long size, const int depth, const int height, const int width,
    const int search_radius, const int patch_radius, const double h2)
{
    const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    const long long plane_size = (long long)height * width;
    const int x = index % width;
    const int y = (index / width) % height;
    const int z = index / plane_size;

    // Self weight one guarantees a defined result even if all weights vanish.
    double weighted_sum = input[index];
    double weight_sum = 1.0;
    float best_weights[K > 0 ? K : 1];
    float best_values[K > 0 ? K : 1];
    int count = 0;

    // Compile away z searches and z patch comparisons in slice-wise mode.
    const int dz_begin = FULL_3D ? -min(search_radius, z) : 0;
    const int dz_end = FULL_3D ? min(search_radius, depth - 1 - z) : 0;
    for (int dx = -min(search_radius, x);
         dx <= min(search_radius, width - 1 - x); ++dx) {
        const int qx = x + dx;
        for (int dy = -min(search_radius, y);
             dy <= min(search_radius, height - 1 - y); ++dy) {
            const int qy = y + dy;
            for (int dz = dz_begin; dz <= dz_end; ++dz) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                const int qz = z + dz;
                const int pz_begin = FULL_3D ? -min(patch_radius, min(z, qz)) : 0;
                const int pz_end = FULL_3D ? min(patch_radius, depth - 1 - max(z, qz)) : 0;
                float distance = 0.0f;
                for (int px = -min(patch_radius, min(x, qx));
                     px <= min(patch_radius, width - 1 - max(x, qx)); ++px) {
                    const float gx = gaussian[px + patch_radius];
                    for (int py = -min(patch_radius, min(y, qy));
                         py <= min(patch_radius, height - 1 - max(y, qy)); ++py) {
                        const float gxy = gx * gaussian[py + patch_radius];
                        for (int pz = pz_begin; pz <= pz_end; ++pz) {
                            const float gaussian_weight = FULL_3D
                                ? gxy * gaussian[pz + patch_radius] : gxy;
                            const float difference =
                                input[(long long)(z + pz) * plane_size +
                                      (long long)(y + py) * width + x + px] -
                                input[(long long)(qz + pz) * plane_size +
                                      (long long)(qy + py) * width + qx + px];
                            distance += gaussian_weight * difference * difference;
                        }
                    }
                }
                const float weight = (float)exp(-(double)distance / h2);
                const float value = input[(long long)qz * plane_size + (long long)qy * width + qx];
                if constexpr (K == 0) {
                    weighted_sum += (double)weight * value;
                    weight_sum += weight;
                } else {
                    // Stable selection; ties retain x, then y, then z order.
                    if (count == K && weight <= best_weights[K - 1]) continue;
                    int position = count < K ? count++ : K - 1;
                    while (position > 0 && weight > best_weights[position - 1]) {
                        best_weights[position] = best_weights[position - 1];
                        best_values[position] = best_values[position - 1];
                        --position;
                    }
                    best_weights[position] = weight;
                    best_values[position] = value;
                }
            }
        }
    }
    if constexpr (K > 0) {
        for (int neighbour = 0; neighbour < count; ++neighbour) {
            weighted_sum += (double)best_weights[neighbour] * best_values[neighbour];
            weight_sum += best_weights[neighbour];
        }
    }
    output[index] = (float)(weighted_sum / weight_sum);
}
