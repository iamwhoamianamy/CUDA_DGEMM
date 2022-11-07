#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>

#include <thrust/host_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include <thrust/device_vector.h>

#include "cuda_kernel_callers.cuh"

namespace thrust
{
	template <class t>
	using pinned_allocator = thrust::system::cuda::experimental::pinned_allocator<t>;
}

template <class func>
std::chrono::milliseconds get_execution_time(func function)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
	function();
	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);
	
	float time = 0;
	cudaEventElapsedTime(&time, start, stop);

	return std::chrono::milliseconds(static_cast<size_t>(time));
}

void print_matrix(const std::vector<real>& matrix, int height, int width)
{
	for (size_t i = 0; i < height; i++)
	{
		for (size_t j = 0; j < width; j++)
		{
			std::cout << get_elem(matrix.data(), i, j, height) << " ";
		}

		std::cout << std::endl;
	}
}

real get_random_normed()
{
	return static_cast<real>(rand()) / static_cast<real>(RAND_MAX);
}

const size_t M = 3000;
const size_t N = 2000;
const size_t K = 4000;

const size_t a_size = M * K * sizeof(real);
const size_t b_size = K * N * sizeof(real);
const size_t c_size = M * N * sizeof(real);

template <template <class> class allocator>
real test_cuda_simple_dgemm()
{
	thrust::host_vector<real, allocator<real>> a(M * K);
	thrust::host_vector<real, allocator<real>> b(K * N);
	thrust::host_vector<real, allocator<real>> c_my(M * N);
	thrust::host_vector<real, allocator<real>> c_cublas(M * N);

	std::generate(a.begin(), a.end(), get_random_normed);
	std::generate(b.begin(), b.end(), get_random_normed);

	thrust::device_vector<real> dev_a(a);
	thrust::device_vector<real> dev_b(b);
	thrust::device_vector<real> dev_c_my(M * N);
	thrust::device_vector<real> dev_c_cublas(M * N);

	cuda_simple_dgemm(dev_a.data().get(), dev_b.data().get(), dev_c_my.data().get(), M, N, K);
	cudaDeviceSynchronize();

	cublas_dgemm(dev_a.data().get(), dev_b.data().get(), dev_c_cublas.data().get(), M, N, K);
	cudaDeviceSynchronize();

	thrust::copy(dev_c_my.begin(), dev_c_my.end(), c_my.begin());
	thrust::copy(dev_c_cublas.begin(), dev_c_cublas.end(), dev_c_cublas.begin());

	real diff = 0;

	for (size_t i = 0; i < c_my.size(); i++)
	{
		diff += std::abs(c_my[i] - c_cublas[i]);
	}

	return diff / c_my.size();
}

template <template <class> class allocator>
std::chrono::milliseconds task()
{
	thrust::host_vector<real, allocator<real>> a(M * K);
	thrust::host_vector<real, allocator<real>> b(K * N);
	thrust::host_vector<real, allocator<real>> c(M * N); 

	std::generate(a.begin(), a.end(), get_random_normed);
	std::generate(b.begin(), b.end(), get_random_normed);

	thrust::device_vector<real> dev_a(M * K);
	thrust::device_vector<real> dev_b(K * N);
	thrust::device_vector<real> dev_c(M * N);

	auto test_function = [&]()
	{
		thrust::copy(a.begin(), a.end(), dev_a.begin());
		thrust::copy(b.begin(), b.end(), dev_b.begin());

		cuda_simple_dgemm(dev_a.data().get(), dev_b.data().get(), dev_c.data().get(), M, N, K);

		thrust::copy(dev_c.begin(), dev_c.end(), c.begin());
	};

	auto execution_time = get_execution_time(test_function); 

	return execution_time;
}

int main(int agrc, char** argv)
{
	std::cout << "Task #1 exec time: " << task<std::allocator>().count() / 1000.0;
	std::cout << "ms" << std::endl;
	std::cout << "Average error: " << test_cuda_simple_dgemm<std::allocator>() << std::endl;

	std::cout << "Task #2 exec time: " << task<thrust::pinned_allocator>().count() / 1000.0;
	std::cout << "ms" << std::endl;
	std::cout << "Average error: " << test_cuda_simple_dgemm<thrust::pinned_allocator>() << std::endl;

	return 0;
}