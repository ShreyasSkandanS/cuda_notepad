#include "/home/shreyas/CUDA_BY_EXAMPLE/common/book.h"
#include "/home/shreyas/CUDA_BY_EXAMPLE/common/cpu_bitmap.h"
#include "/home/shreyas/CUDA_BY_EXAMPLE/common/cpu_anim.h"
#include "cuda.h"

#define DIM 1024
#define PI 3.14159f

__global__ void pixelmapping(unsigned char *ptr)
{
	// Find linearized pixel co-ordinate locations (thread index in 2D)
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	// Allocate 16x16 array of shared memory
	__shared__ float shared[16][16];
	
	const float period = 128.0f;

	// "each thread will be computing a pixel value for a single output location"
	shared[threadIdx.x][threadIdx.y] = 255*(sinf(x*2.0f*PI/period) + 1.0f)*(sinf(y*2.0f*PI/period))/4.0f;

	// Sync between write and read is necessary to generate correct blob pixels
	__syncthreads();

	// Display the image with the GREEN channel being a function of the given colour mapping
	ptr[offset*4 + 0] = 0;
	ptr[offset*4 + 1] = 0;
	ptr[offset*4 + 2] = shared[15-threadIdx.x][15-threadIdx.y];
	ptr[offset*4 + 3] = 255;
}

int main(void)
{	
	CPUBitmap bitmap(DIM,DIM);
	unsigned char *device_bitmap;
	
	cudaMalloc((void**)&device_bitmap,bitmap.image_size());
	dim3 grids(DIM/16,DIM/16);
	dim3 threads(16,16);
	
	pixelmapping<<<grids,threads>>>(device_bitmap);

	cudaMemcpy(bitmap.get_ptr(),device_bitmap,bitmap.image_size(),cudaMemcpyDeviceToHost);
	bitmap.display_and_exit();
	cudaFree(device_bitmap);
}


