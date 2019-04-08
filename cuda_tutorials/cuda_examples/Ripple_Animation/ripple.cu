#include "/home/shreyas/CUDA_BY_EXAMPLE/common/book.h"
#include "/home/shreyas/CUDA_BY_EXAMPLE/common/cpu_bitmap.h"
#include "/home/shreyas/CUDA_BY_EXAMPLE/common/cpu_anim.h"

// Use NVCC compiler as shown to link OpenGL libraries:
// 	nvcc -o ripple.o ripple.cu -lGL -lglut

#define DIM 640

// Create a structure for bitmap object
struct Data
{
	unsigned char *device_bitmap;
	CPUAnimBitmap *bitmap;
};

// Create a function to free up GPU memory after task completion
void garbagecollection(Data *d)
{
	cudaFree(d->device_bitmap);
}

// DEVICE function to generate pixel colour values at every (x,y) location 
// based on some mathematical function to generate a ripple effect while 
// animating over time t (ticks)
__global__ void pixel_mapping( unsigned char *ptr, int ticks)
{
	// Establish (x,y) location scheme based on thread and block dimensions
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	// Linearize (x,y) dimensions
	int incr = x + y * blockDim.x * gridDim.x;

	float px = x - DIM/2;
	float py = y - DIM/2;
	float euc_dist = sqrt((px*px)+(py*py));
	
	// Mathematical function to obtain pixel value at each location
	unsigned char clr = (unsigned char)( 128.0f + 127.0f * (cos(euc_dist/10.0f-ticks/7.0f)) / (euc_dist/10.0f + 1.0f) );
	
	// Apply pixel value to individual colour channels 
	ptr[incr*4 + 0] = clr;
	ptr[incr*4 + 1] = clr;
	ptr[incr*4 + 2] = 255;
	ptr[incr*4 + 3] = 255;
}

void generate_frame(Data *d, int ticks)
{
	// Divide the full image into blocks and threads
	dim3 blocks(DIM/16,DIM/16);
	dim3 threads(16,16);
	// Map colour values to bitmap on the GPU
	pixel_mapping<<<blocks,threads>>>(d->device_bitmap,ticks);
	// Copy the resulting bitmap to HOST
	cudaMemcpy(d->bitmap->get_ptr(),d->device_bitmap,d->bitmap->image_size(),cudaMemcpyDeviceToHost);
}

int main(void)
{
	Data data;
	CPUAnimBitmap bitmap(DIM,DIM,&data);
	data.bitmap = &bitmap;
	cudaMalloc((void**)&data.device_bitmap,bitmap.image_size());
	bitmap.anim_and_exit((void (*)(void*,int))generate_frame,(void(*)(void*))garbagecollection);
}

