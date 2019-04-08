#include "/home/shreyas/CUDA_BY_EXAMPLE/common/book.h"

// Define a macro for finding the smaller of two integers
#define min_value(a,b) (a<b?a:b)
const int threads_per_block = 256;
const int N = 33 * 1024;
const int blocks_per_grid = min_value(32,(N+threads_per_block-1)/threads_per_block);

__global__ void dotprod(float *vector1, float *vector2, float *result)
{ 
	// Create a buffer on the GPU shared memory to store the intermediate sums
	__shared__ float buffer[threads_per_block];
	// Find thread index using block offset
	int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
	// The buffer index will be the same as the thread index since each thread 
	// will calculate it's own sum and that sum will be stored in the shared 
	// memory
	int buffer_idx = threadIdx.x;
	float temp_sum = 0;
	while (thread_id < N)
	{
		// Calculate the intermediate products and store in a temp variable
		temp_sum = temp_sum + vector1[thread_id]*vector2[thread_id];
		// Increment thread element index by total number of elements in thread
		thread_id = thread_id + blockDim.x * gridDim.x;
	}
	// Store this temporary sum of products (running sum) in the shared memory buffer
	buffer[buffer_idx] = temp_sum;

	// Point of synchronisation for all above threaded computations
	__syncthreads();

	// REDUCTIONS -- Summation Reduction --
	// Since we have 256 threads, to reduce this completely using two thread sum,
	// we will need x:2^x=256 iterations (x=8 iterations)
	int i = blockDim.x/2;
	while (i!=0)
	{
		if(buffer_idx < i)
		{
			// Add i'th element of each thread in buffer
			buffer[buffer_idx] += buffer[buffer_idx+i];
		}
		__syncthreads();
		// i then reduces by half since 256 elements would become 128 after 
		// the first reduction step
		i = i/2;
	}
	if(buffer_idx == 0)
	{
		// Add thread sum to block result, return control to CPU to perform
		// final arithmetic addition of block sums
		result[blockIdx.x] = buffer[0];
	}
}

int main(void)
{
	float *vec1,*vec2,dot_res,*block_sum;
	float *device_vec1,*device_vec2,*device_block_sum;
	vec1 = (float*)malloc(N*sizeof(float));
	vec2 = (float*)malloc(N*sizeof(float));
	block_sum = (float*)malloc(blocks_per_grid*sizeof(float));
	
	cudaMalloc((void**)&device_vec1,N*sizeof(float));
	cudaMalloc((void**)&device_vec2,N*sizeof(float));
	cudaMalloc((void**)&device_block_sum,blocks_per_grid*sizeof(float));

	for (int i=0;i<N;i++)
	{
		vec1[i] = i;
		vec2[i] = i*2;
	}

	cudaMemcpy(device_vec1,vec1,N*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(device_vec2,vec2,N*sizeof(float),cudaMemcpyHostToDevice);
	
	dotprod<<<blocks_per_grid,threads_per_block>>>(device_vec1,device_vec2,device_block_sum);

	cudaMemcpy(block_sum,device_block_sum,blocks_per_grid*sizeof(float),cudaMemcpyDeviceToHost);

	// Now that we have the block sum results, we continue with rest of the computation on the CPU
	dot_res = 0;
	for (int i=0;i<blocks_per_grid;i++)
	{
		dot_res = dot_res + block_sum[i];
	}

	// Verify the dot product result by computing the closed form solution
	#define sum_of_squares(x) (x*(x+1)*(2*x+1)/6)
	
	printf("%.6g == %.6g?\n",dot_res,2*sum_of_squares((float)(N-1)));

	cudaFree(device_vec1);
	cudaFree(device_vec2);
	cudaFree(device_block_sum);
	
	free(vec1);
	free(vec2);
	free(block_sum);
}

	
	
	
			
