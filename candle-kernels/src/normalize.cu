#include "cuda_utils.cuh"

extern "C" __global__ void normalize_f32( 
    const size_t numel, 
    float *lhs, 
    const size_t size, 
    const float epsilon
) { 
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i >= numel) { 
        return; 
    } 

    const size_t offset = i * size;

    float var = 0.0;
    for (int i=0; i < size; i ++){
        var += lhs[offset + i] * lhs[offset + i];
    }
    var /= size;
    var += epsilon;
    const float std = sqrt(var);
    for (int i=0; i < size; i ++){
        lhs[offset + i] /= std;
    }
} 

extern "C" __global__ void normalize_f16( 
    const size_t numel, 
    __half *lhs, 
    const size_t size, 
    const float epsilon,
    float *var
) { 
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
        size_t row = i / size;
        float tmp =  __half2float(lhs[i]) * __half2float(lhs[i]) / size;
        atomicAdd(var + row, tmp);
    }
    __syncthreads();
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
        size_t row = i / size;

        lhs[i] = __float2half( __half2float(lhs[i]) / sqrt(var[row] + epsilon));
    }
} 

