#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <chrono>


cudaError_t multiply_matrix(float* arr1, float* arr2, float* res, unsigned int dims, unsigned int n);

void init_matrix(float* arr1, float* arr2, unsigned int n) {
	for (int i = 0; i < n * n; i++) {
		arr1[i] = rand() % 10;
		arr2[i] = rand() % 10;
	}
}

__global__ void matrix_multiply_kernel(float* arr1, float* arr2, float* res, unsigned int n) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < n && col < n) {
		float mul = 0.0;
		
		for (int i = 0; i < n; i++)
			mul += arr1[row * n + i] * arr2[i * n + col];

		res[row * n + col] = mul;
	}

}


//function to print the GPU properties
void print_cuda_device_properties(int deviceId) {
	cudaDeviceProp device_props;
	cudaGetDeviceProperties(&device_props, deviceId);

	fprintf(stdout, "<----- GPU INFO ----->\n\n");
	fprintf(stdout, "\tDevice %d: %s\n", deviceId, device_props.name);
	fprintf(stdout, "\tTotal global memory: %.2f MB\n", (float)device_props.totalGlobalMem / (1024 * 1024));
	fprintf(stdout, "\tShared memory per block: %d bytes\n", device_props.sharedMemPerBlock);
	fprintf(stdout, "\tMaximum threads per block: %d\n", device_props.maxThreadsPerBlock);
	fprintf(stdout, "\tMaximum grid size: (%d, %d, %d)\n", device_props.maxGridSize[0], device_props.maxGridSize[1], device_props.maxGridSize[2]);
	fprintf(stdout, "\n<----- END OF GPU INFO ----->\n\n");
}

int main() {

	// Get the GPU properties
	print_cuda_device_properties(0);

	// Initilize the random seed
	srand(time(NULL));

	cudaError_t cuda_status;

	const unsigned int n = 2048;
	size_t dims = n * n * sizeof(float);

	// Allocate memory for the host
	float* h_arr1 = new float[n * n];
	float* h_arr2 = new float[n * n];
	float* h_res = new float[n * n];

	// Initilize the matrix values 
	init_matrix(h_arr1, h_arr2, n);

	// Call the multiply_matrix function
	cuda_status = multiply_matrix(h_arr1, h_arr2, h_res, dims, n);
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "[!] multiply_matrix Failed!\n");
		goto _EndOfFunc;
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cuda_status = cudaDeviceReset();
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		goto _EndOfFunc;
	}

_EndOfFunc:

	// Free the resources
	delete[] h_arr1;
	delete[] h_arr2;
	delete[] h_res;

	return 0;

}

cudaError_t multiply_matrix(float* arr1, float* arr2, float* res, unsigned int dims, unsigned int n) {
	float* dev_arr1 = 0;
	float* dev_arr2 = 0;
	float* dev_res = 0;

	cudaError_t cuda_status;

	// Choose which GPU to run on
	cuda_status = cudaSetDevice(0);
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "[!] cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	//Allocate memory on the GPU
	cuda_status = cudaMalloc(&dev_arr1, dims);
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "[!] Cuda Malloc [arr1] Failed\n");
		goto Error;
	}

	cuda_status = cudaMalloc(&dev_arr2, dims);
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "[!] Cuda Malloc [arr2] Failed\n");
		goto Error;
	}

	cuda_status = cudaMalloc(&dev_res, dims);
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "[!] Cuda Malloc [res] Failed\n");
		goto Error;
	}

	// Copy input matrices to the device
	cuda_status = cudaMemcpy(dev_arr1, arr1, dims, cudaMemcpyHostToDevice);
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "[!] cudaMemcpy [arr1] Failed\n");
		goto Error;
	}

	cuda_status = cudaMemcpy(dev_arr2, arr2, dims, cudaMemcpyHostToDevice);
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "[!] cudaMemcpy [arr2] Failed\n");
		goto Error;
	}

	//Set up grid and block size dimensions
	dim3 block_size(32, 32);
	dim3 grid_size((n + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y);

	
	//Execute the cuda kernal
	matrix_multiply_kernel << <grid_size, block_size >> > (dev_arr1, dev_arr2, dev_res, n);
	//Setup a start timer
	auto start = std::chrono::high_resolution_clock::now();

	// Check for any errors launching the kernel
	cuda_status = cudaGetLastError();
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "[!] addKernel launch failed with error: %s\n", cudaGetErrorString(cuda_status));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.
	cuda_status = cudaDeviceSynchronize();
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "[!] cudaDeviceSynchronize Failed with error code %d\n", cuda_status);
		goto Error;
	}

	// End the timer
	auto end = std::chrono::high_resolution_clock::now();

	// Copy the result from the GPU to the host (memory)
	cuda_status = cudaMemcpy(res, dev_res, dims, cudaMemcpyDeviceToHost);
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "[!] cudaMemcpy [res] Failed!\n");
		goto Error;
	}

	// Print the execution time
	fprintf(stdout ,"\nMatrix multiplication took: %lld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

Error:
	// Free the resources
	cudaFree(dev_arr1);
	cudaFree(dev_arr2);
	cudaFree(dev_res);
	
	return cuda_status;
}