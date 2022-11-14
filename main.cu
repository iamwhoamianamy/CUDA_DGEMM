#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>

#include <thrust/host_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/inner_product.h>

#include "cuda_kernel_callers.cuh"

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

real get_random_normed()
{
	return static_cast<real>(rand()) / static_cast<real>(RAND_MAX);
}

template <class host_allocator>
real test_cuda_simple_dgemm_error(size_t m, size_t n, size_t k)
{
	thrust::host_vector<real, host_allocator> a(m * k);
	thrust::host_vector<real, host_allocator> b(k * n);
	thrust::host_vector<real, host_allocator> c_my(m * n);
	thrust::host_vector<real, host_allocator> c_cublas(m * n);

	std::generate(a.begin(), a.end(), get_random_normed);
	std::generate(b.begin(), b.end(), get_random_normed);

	thrust::device_vector<real> dev_a(a);
	thrust::device_vector<real> dev_b(b);
	thrust::device_vector<real> dev_c_my(m * n, 0);
	thrust::device_vector<real> dev_c_cublas(m * n, 0);

	cuda_simple_dgemm(dev_a.data().get(), dev_b.data().get(), dev_c_my.data().get(), m, n, k);
	cudaDeviceSynchronize();

	cublas_dgemm(dev_a.data().get(), dev_b.data().get(), dev_c_cublas.data().get(), m, n, k);
	cudaDeviceSynchronize();

	thrust::copy(dev_c_my.begin(), dev_c_my.end(), c_my.begin());
	thrust::copy(dev_c_cublas.begin(), dev_c_cublas.end(), c_cublas.begin());

	real diff = 0;

	for (size_t i = 0; i < c_my.size(); i++)
	{
		diff += std::abs(c_my[i] - c_cublas[i]);
	}

	return thrust::inner_product(
		dev_c_my.begin(),
		dev_c_my.end(), 
		dev_c_cublas.begin(), 
		real(0), 
		thrust::plus<real>(),
		thrust::minus<real>());
}

template <class host_allocator, class algorithm>
std::chrono::milliseconds multiply_matrices(size_t m, size_t n, size_t k, algorithm alg)
{
	thrust::host_vector<real, host_allocator> a(m * k);
	thrust::host_vector<real, host_allocator> b(k * n);
	thrust::host_vector<real, host_allocator> c(m * n); 

	std::generate(a.begin(), a.end(), get_random_normed);
	std::generate(b.begin(), b.end(), get_random_normed);

	thrust::device_vector<real> dev_a(m * k);
	thrust::device_vector<real> dev_b(k * n);
	thrust::device_vector<real> dev_c(m * n);

	auto test_function = [&]()
	{
		thrust::copy(a.begin(), a.end(), dev_a.begin());
		thrust::copy(b.begin(), b.end(), dev_b.begin());

		alg(dev_a.data().get(), dev_b.data().get(), dev_c.data().get(), m, n, k);
		cudaDeviceSynchronize();

		thrust::copy(dev_c.begin(), dev_c.end(), c.begin());
	};

	return get_execution_time(test_function);
}

namespace thrust
{
	template <class t>
	using pinned_allocator = thrust::system::cuda::experimental::pinned_allocator<t>;
}

void run_single_test(size_t m, size_t n, size_t k)
{
	auto simple_default_time = multiply_matrices<std::allocator<real>>(m, n, k, cuda_simple_dgemm).count() / 1000.0;
	auto simple_pinned_time = multiply_matrices<thrust::pinned_allocator<real>>(m, n, k, cuda_simple_dgemm).count() / 1000.0;
	auto cublas_default_time = multiply_matrices<std::allocator<real>>(m, n, k, cublas_dgemm).count() / 1000.0;
	auto cublas_pinned_time = multiply_matrices<thrust::pinned_allocator<real>>(m, n, k, cublas_dgemm).count() / 1000.0;

	std::cout << m * n * k << " ";
	std::cout << simple_default_time << " ";
	std::cout << simple_pinned_time << " ";
	std::cout << cublas_default_time << " ";
	std::cout << cublas_pinned_time << std::endl;
}

void run_multiple_tests(size_t count)
{
	size_t m = 1000;
	size_t n = 1000;
	size_t k = 1000;

	const double factor = std::pow(static_cast<double>(2), 1.0 / 3);

	for (size_t i = 0; i < count; i++)
	{
		m *= factor;
		n *= factor;
		k *= factor;

		run_single_test(m, n, k);
	}
}

int main(int argc, char** argv)
{
	switch(argc)
	{
		case 2:
		{
			size_t count;

			try
			{
				count = std::atoll(argv[1]);
				run_multiple_tests(count);
			}
			catch(const std::exception& e)
			{
				std::cout << "Error on params parsing!" << std::endl;
				return 1;
			}

			break;
		}
		case 4:
		{
			try
			{
				size_t m = std::atoll(argv[1]);
				size_t n = std::atoll(argv[2]);
				size_t k = std::atoll(argv[3]);
				run_single_test(m, n, k);
			}
			catch(const std::exception& e)
			{
				std::cout << "Error on params parsing!" << std::endl;
				return 1;
			}

			break;
		}
		default:
		{
			std::cout << "Provide:" << std::endl;
			std::cout << "	1 param - count of tests" << std::endl;
			std::cout << "or:" << std::endl;
			std::cout << "	3 params - m n k" << std::endl;

			return 1;
		}
	}

	return 0;
}