#include "/home/shreyas/CUDA_BY_EXAMPLE/common/book.h"
#include <stdio.h>

// Block limit is 65,535 ; Thread limit is 512

#define N (25*256)

__global__ void addvec_threads(int *vec1, int *vec2, int *sum)
{
	// Indexing scheme for multiple block and multiple thread setup
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < N)
	{
		// Addition of vectors is defined as before
		sum[tid] = vec1[tid] + vec2[tid];
		// Thread increment is a function of the block and grid size
		// NOTE:
		// 	"We want each parallel thread to start on a different
		// data index, so we just need to take our thread and block
		// indexes and linearize them.."
		//
		// We can now work more elements than restricted by the hard
		// -ware limit of 65,535 * thread_count (<512)
		tid = tid + blockDim.x * gridDim.x;
	}
}


int main(void)
{
	FILE *fp;
	char output[] = "thread_block_output.txt";

	int vec1[N];
	int vec2[N];
	int sum[N];
	int *device_v1, *device_v2, *device_sum;
	
	cudaMalloc((void**)&device_v1, N*sizeof(int));
	cudaMalloc((void**)&device_v2, N*sizeof(int));
	cudaMalloc((void**)&device_sum, N*sizeof(int));
	
	for (int iter=0;iter<N;iter++)
	{
		vec1[iter] = iter;
		vec2[iter] = iter*iter;
	}
	
	// CUDA MEMCPY format (destination,source,size,transfer_path)
	// Prepare parameters for DEVICE computation
	cudaMemcpy(device_v1,vec1,N*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(device_v2,vec2,N*sizeof(int),cudaMemcpyHostToDevice);
	// Perform parallel computation on DEVICE
	addvec_threads<<<128,128>>>(device_v1,device_v2,device_sum);
	// Prepare result for output display on HOST
	cudaMemcpy(sum, device_sum, N*sizeof(int), cudaMemcpyDeviceToHost);

	// Write vector sum output to a FILE
	fp = fopen(output,"w+");
	for (int ii = 0; ii < N; ii++)
	{
	fprintf(fp,"%d\r\n",sum[ii]);
	}
	fclose(fp);
	
	cudaFree(device_v1);
	cudaFree(device_v2);
	cudaFree(device_sum);
	return 0;
	
}
	
