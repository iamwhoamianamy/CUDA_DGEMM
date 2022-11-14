#include <cmath>

#include "cuda_kernels.cuh"

void cuda_simple_dgemm(
   const real* a, const real* b, real* c,
   size_t m, size_t n, size_t k);

void cublas_dgemm(
   const real* a, const real* b, real* c,
   size_t m, size_t n, size_t k);