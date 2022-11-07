all: cuda main.cpp
	nvcc main.cpp cuda_kernels.o cuda_kernel_callers.o -o main.exe
cuda: cuda_kernels.cu cuda_kernel_callers.cu
	nvcc -c cuda_kernels.cu -o cuda_kernels.o
	nvcc -c cuda_kernel_callers.cu -o cuda_kernel_callers.o
