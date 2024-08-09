#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

__device__ size_t floor_device(float x) {
    return floor(x);
}

size_t computefloor(float x) {
    return floor(x);
}

__device__ float fmod_device(float x, float y) {
    return fmod(x, y);
}

float computeCeil(float num) {
    return ceilf(num);
}

__device__ float ceil_device(float num) {
    return ceilf(num);
}

__global__ void setNegativeToZero(float* restored, size_t rows, size_t cols) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < rows * cols) {
        restored[idx] = (restored[idx] < 0) ? 0 : restored[idx];
    }
}

__global__ void split_lisi(float* data_1, float* data_2, float* result, size_t size, size_t unit_size, size_t ima, size_t unit_num) {

    extern  __shared__  float sharedNumDen[];
    
    size_t bid = blockIdx.x; // tile index
    size_t tid = threadIdx.x;
    
    size_t i_id = floor_device(bid/unit_num);
    size_t j_id = fmod_device(bid,unit_num);
    size_t factor = ceil_device(static_cast<float>(unit_size*unit_size)/1024);
    
    float C1 = 1e-4;
    float C2 = 1e-4;
    float D = C1/2.0;
    size_t I_id;
    size_t J_id;
    float sum1;
    float sub1;
    size_t rows;
    size_t cols;
    
    for (size_t fac = 1; fac <= factor;fac = fac + 1){
		if (tid+(fac-1)*1024 < unit_size*unit_size){
			if (fac == 1) {
				sharedNumDen[tid] = 0; // Numerator
				sharedNumDen[tid+1024] = 0; // Denominator x
				sharedNumDen[tid+2048] = 0; // Denominator y
			}
			rows = floor_device((tid+(fac-1)*1024)/unit_size);
			cols = fmod_device((tid+(fac-1)*1024),unit_size);
			
			I_id = i_id * unit_size + rows;
			J_id = j_id * unit_size + cols;
			sum1 = data_1[I_id * ima + J_id] + data_2[I_id * ima + J_id];
			sub1 = data_1[I_id * ima + J_id] - data_2[I_id * ima + J_id];
			
			sharedNumDen[tid] = sharedNumDen[tid] + abs(sum1)/(abs(sub1)+C1);
			sharedNumDen[tid+1024] = sharedNumDen[tid+1024] + data_1[I_id * ima + J_id];
			sharedNumDen[tid+2048] = sharedNumDen[tid+2048] + data_2[I_id * ima + J_id];
		} else {
			if (fac == 1) {
				sharedNumDen[tid] = 0; // Numerator
				sharedNumDen[tid+1024] = 0; // Denominator x
				sharedNumDen[tid+2048] = 0; // Denominator y
			}
		}
	}
    
    for (size_t d = blockDim.x/2;d>0;d = d/2){
		__syncthreads();
		if (tid<d) {
			sharedNumDen[tid] += sharedNumDen[tid+d];
			sharedNumDen[tid+1024] += sharedNumDen[tid+1024+d];
			sharedNumDen[tid+2048] += sharedNumDen[tid+2048+d];
		}
	}
	
	if (tid==0) {
		result[bid] = D*sharedNumDen[0]/(max(sharedNumDen[1024],sharedNumDen[2048]) + C2);
	}
}

int splitlis(float* input_data_1, float* input_data_2, float* result_array, size_t size, size_t unit_size, size_t ima) {
    
    float* d_data_1;
    float* d_data_2;
    float* result_data;
    cudaError_t cudaStatus;

    size_t ima_pow = ima*ima;
    
    size_t unit_num = ima/unit_size;
    
    cudaMalloc((void**)&d_data_1, size * sizeof(float));
    cudaMalloc((void**)&d_data_2,  size * sizeof(float));
    cudaMalloc((void**)&result_data, unit_num * unit_num * sizeof(float));
    
    cudaMemcpy(d_data_1, input_data_1, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data_2, input_data_2, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(result_data, result_array, unit_num * unit_num * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
    
    cudaEventRecord(start);    
    
    size_t num_threads = 1024;
    size_t num_blocks = computeCeil(static_cast<float>(ima_pow)/num_threads);
    
    setNegativeToZero<<<num_blocks,num_threads>>>(d_data_1, ima, ima);
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error 1 : %s\n", cudaGetErrorString(cudaStatus));
	}
    setNegativeToZero<<<num_blocks,num_threads>>>(d_data_2, ima, ima);
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error 2 : %s\n", cudaGetErrorString(cudaStatus));
	}

    num_threads = 1024;
    num_blocks = unit_num*unit_num;
    size_t shared_mem_size = 3 * num_threads * sizeof(float);
    split_lisi<<<num_blocks,num_threads,shared_mem_size>>>(d_data_1, d_data_2, result_data, size, unit_size, ima, unit_num);
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error 3 : %s\n", cudaGetErrorString(cudaStatus));
	}
	
	cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "Time elapsed: " << milliseconds << " ms" << std::endl;
	
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaMemcpy(result_array, result_data, unit_num * unit_num * sizeof(float), cudaMemcpyDeviceToHost);
	cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error 4 : %s\n", cudaGetErrorString(cudaStatus));
	}

    cudaFree(d_data_1);
    cudaFree(d_data_2);
    cudaFree(result_data);
    return 0;
}
