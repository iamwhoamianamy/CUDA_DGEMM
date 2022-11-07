#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "defines.hpp"

#define __all__ __device__ __host__

inline __all__ int flat_id(int i, int j, int height)
{
	return j * height + i;
}

inline __all__ double get_elem(const real* M, int i, int j, int height)
{
	return M[flat_id(i, j, height)];
}

inline __all__ double get_elem(real* M, int i, int j, int height)
{
	return get_elem(const_cast<const real*>(M), i, j, height);
}

inline __all__ void set_elem(real *M, int i, int j, int height, real val)
{
	M[flat_id(i, j, height)] = val;
}
