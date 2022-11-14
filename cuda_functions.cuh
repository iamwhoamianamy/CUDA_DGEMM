#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "defines.hpp"

#define __all__ __device__ __host__

inline __all__ int flat_id(int i, int j, int height)
{
	return j * height + i;
}

inline __all__ double get_elem(const real* matrix, int i, int j, int height)
{
	return matrix[flat_id(i, j, height)];
}

inline __all__ double get_elem(real* matrix, int i, int j, int height)
{
	return get_elem(const_cast<const real*>(matrix), i, j, height);
}

inline __all__ void set_elem(real *matrix, int i, int j, int height, real val)
{
	matrix[flat_id(i, j, height)] = val;
}