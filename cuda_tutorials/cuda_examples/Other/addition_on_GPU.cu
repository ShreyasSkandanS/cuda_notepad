#include <stdio.h>
#include "/home/shreyas/CUDA_BY_EXAMPLE/common/book.h"

//Size of the vectors
#define N 50

__global__ void add(int num_1, int num_2, int *device_sum)
{
	*device_sum = num_1 + num_2;
}

__global__ void addvec(int *mask1,int *mask2, int *maskadd)
{
	int thread_index = blockIdx.x;
	if (thread_index < N)
	{
	maskadd[thread_index] = mask1[thread_index] + mask2[thread_index];
	}
}

int main(void)
{
	int host_total;
	int user_num_1, user_num_2;
	int *device_total;
	int validate;

	int number_of_cuda_devices;
	cudaGetDeviceCount(&number_of_cuda_devices);

	cudaDeviceProp properties;

	printf("Total number of CUDA capable devices is: %d\n\n",number_of_cuda_devices);

	for(int cuda_dev=0;cuda_dev<number_of_cuda_devices;cuda_dev++)
	{
		cudaGetDeviceProperties(&properties,cuda_dev);
		printf("------ DEVICE %d ------\n",(cuda_dev+1));
		printf("Total global memory available on this device is: %ld bytes\n\n",properties.totalGlobalMem);
	}

	// ADDING TWO SCALARS ON THE GPU - Does not exploit the parallel computation ability of the GPU	
	printf("----- SCALAR ADDITION ON GPU -----\n");
	printf("Enter the first number:\n");
	scanf("%d",&user_num_1);
	printf("Enter the second number:\n");
	scanf("%d",&user_num_2);

	cudaMalloc((void**)&device_total,sizeof(int));

	add<<<1,1>>>(user_num_1,user_num_2,device_total);

	cudaMemcpy(&host_total,device_total,sizeof(int),cudaMemcpyDeviceToHost);

	printf("The sum of the two numbers is : %d\n",host_total);
	cudaFree(device_total);


	// ADDING TWO VECTORS ON THE GPU EXPLOITING PARALLEL COMPUTATION
	printf("----- VECTOR ADDITION ON GPU -----\n");
	// Assuming that our host system already contains two vectors (such as two unrolled image mask vectors)
	// and we wish to perform a mask addition operation
	int mask_1[N], mask_2[N], mask_addition[N];
	int *mask1_device, *mask2_device, *maskaddition_device;

	cudaMalloc((void**)&mask1_device, N*sizeof(int));
	cudaMalloc((void**)&mask2_device, N*sizeof(int));
	cudaMalloc((void**)&maskaddition_device, N*sizeof(int));

	// Fill in vectors a and b with some arbitrary values
	for (int tmpidx=0;tmpidx<N;tmpidx++)
	{
	mask_1[tmpidx] = 1;
	mask_2[tmpidx] = 0;
	}


	cudaMemcpy(mask1_device,mask_1,N*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(mask2_device,mask_2,N*sizeof(int),cudaMemcpyHostToDevice);
	
	addvec<<<N,1>>>(mask1_device,mask2_device,maskaddition_device);

	cudaMemcpy(mask_addition,maskaddition_device,N*sizeof(int),cudaMemcpyDeviceToHost);

	validate = 0;
	// Print out the computed sum of vectors
	for (int tempidx=0;tempidx<N;tempidx++)
	{
	validate = validate + mask_addition[tempidx];
	}
	printf("Validation of vector sum - Total: %d | Input: %d\n",validate,N);
	
	cudaFree(mask1_device);
	cudaFree(mask2_device);
	cudaFree(maskaddition_device);

	return 0;
}
