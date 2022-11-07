#include <cmath>

#include "cuda_kernels.cuh"

 void cuda_simple_dgemm(
   const real* a, const real* b, real* c,
   size_t M, size_t N, size_t K);

   void cublas_dgemm(
   const real* a, const real* b, real* c,
   size_t M, size_t N, size_t K);