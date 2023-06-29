#include "cuda_utils.cuh"

extern "C" __global__ void normalize_f32( 
    const size_t numel, 
    const float *lhs, 
    float *rhs, 
    const size_t size, 
    const float epsilon
) { 
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i >= numel) { 
        return; 
    } 

    const size_t offset = i * size;

    // float sum = 0.0;
    // for (int i=0; i < size; i ++){
    //     sum += lhs[offset + i];
    // }
    // sum /= size;
    // for (int i=0; i < size; i ++){
    //     lhs[offset + i] -= sum;
    // }

    float var = 0.0;
    for (int i=0; i < size; i ++){
        var += lhs[offset + i] * lhs[offset + i];
    }
    // var /= size;
    var += epsilon;
    const float std = sqrt(var);
    for (int i=0; i < size; i ++){
        rhs[offset + i] = lhs[offset + i] / std;
    }
} 

extern "C" __global__ void normalize_f16( 
    const size_t numel, 
    const __half *lhs, 
    __half *rhs, 
    const size_t size, 
    const float epsilon
) { 
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i >= numel) { 
        return; 
    } 

    const size_t offset = i * size;

    // float sum = 0.0;
    // for (int i=0; i < size; i ++){
    //     sum +=  __half2float(lhs[offset + i]);
    // }
    // sum /= size;
    // for (int i=0; i < size; i ++){
    //     lhs[offset + i] -= __half2float(sum);
    // }

    float var = 0.0;
    for (int i=0; i < size; i ++){
        var +=  __half2float(lhs[offset + i]) * __half2float(lhs[offset + i]);
    }
    // var /= size;
    var += epsilon;
    const float std = sqrt(var);
    for (int i=0; i < size; i ++){
        // lhs[offset + i] /= __float2half(std);
        rhs[offset + i] = lhs[offset + i] / __float2half(std);
    }
} 

