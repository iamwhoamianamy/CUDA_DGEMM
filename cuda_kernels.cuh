#include "cuda_functions.cuh"

__global__ void cuda_simple_dgemm_kernel(
   const real* a, const real* b, real* c,
   size_t M, size_t N, size_t K);