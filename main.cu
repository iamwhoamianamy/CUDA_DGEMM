#include <iostream>
#include <vector>
#include <algorithm>

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

typedef float real;

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

__global__ void cuda_simple_dgemm_kernel(
   const real* a, const real* b, real* c,
   size_t M, size_t N, size_t K)
{
	real sum = 0;
	const size_t i = blockDim.y * blockIdx.y + threadIdx.y;
	const size_t j = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < M && j < N)
	{
		for (size_t k = 0; k < K; k++)
		{
			sum += get_elem(a, i, k, M) * get_elem(b, k, j, K);
		}

		set_elem(c, i, j, M, sum);
	}
}

inline void cuda_simple_dgemm(
   const real* a, const real* b, real* c,
   size_t M, size_t N, size_t K)
{
	const size_t thread_per_block_1_dim = 32;
	dim3 thread_per_block(thread_per_block_1_dim, thread_per_block_1_dim);	
	dim3 blocks(
		std::ceil(static_cast<float>(M) / thread_per_block_1_dim),
		std::ceil(static_cast<float>(N) / thread_per_block_1_dim));

	cuda_simple_dgemm_kernel<<<blocks, thread_per_block>>>(a, b, c, M, N, K);
}

template <class func>
float get_execution_time(func function)
{
	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	function();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	return time;
}

void print_matrix(const std::vector<real>& M, int height, int width)
{
	for (size_t i = 0; i < height; i++)
	{
		for (size_t j = 0; j < width; j++)
		{
			std::cout << get_elem(M.data(), i, j, height) << " ";
		}

		std::cout << std::endl;
	}
}

real get_random_normed()
{
	return static_cast<real>(rand()) / static_cast<real>(RAND_MAX);
}

const size_t M = 3;
const size_t N = 2;
const size_t K = 4;

const size_t a_size = M * K * sizeof(real);
const size_t b_size = K * N * sizeof(real);
const size_t c_size = M * N * sizeof(real);

void task_1()
{
	std::vector<real> a(M * K);
	std::vector<real> b(K * N);
	std::vector<real> c(M * N);

	std::generate(a.begin(), a.end(), get_random_normed);
	std::generate(b.begin(), b.end(), get_random_normed);

	real* dev_a;
	real* dev_b;
	real* dev_c;

	cudaMalloc((void**)&dev_a, a_size);
	cudaMalloc((void**)&dev_b, b_size);
	cudaMalloc((void**)&dev_c, c_size);

	auto test_function = [&]()
	{
		cudaMemcpy(dev_a, a.data(), a_size, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_b, b.data(), b_size, cudaMemcpyHostToDevice);

		cuda_simple_dgemm(dev_a, dev_b, dev_c, M, N, K);

		cudaMemcpy(c.data(), dev_c, c_size, cudaMemcpyDeviceToHost);
	};

	std::cout << "Task #1 exec time: " << get_execution_time(test_function) << std::endl;
	std::cout << "Result matrix: " << std::endl;
	print_matrix(c, M, N);
}

int main(int agrc, char** argv)
{
	task_1();

	return 0;
}