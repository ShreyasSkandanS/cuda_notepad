#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"
#include "bitmap_image.hpp"

#include <stdio.h>

const double PI = 3.14159265358979323846;

const int BLOCKDIM = 32;

// Note: this must be an odd number
__device__ const int FILTER_SIZE = 15;
__device__ const int FILTER_HALFSIZE = FILTER_SIZE >> 1;

__device__ const int BLUE_MASK = 0x00ff0000;
__device__ const int GREEN_MASK = 0x0000ff00;
__device__ const int RED_MASK = 0x000000ff;

__device__ int index(int x, int y, const cudaPitchedPtr& cpp) 
{
	return (y * (cpp.pitch / 4)) + x;
}


__device__ int clamp(int value, int bound) 
{
	if (value < 0) {
		return 0;
	}
	if (value < bound) {
		return value;
	}
	return bound - 1;
}


__global__ void blurGlobal(cudaPitchedPtr src, cudaPitchedPtr dst, float* gaussian) 
{
	int x = (blockDim.x * blockIdx.x) + threadIdx.x;
	int y = (blockDim.y * blockIdx.y) + threadIdx.y;

	float r = 0.0, g = 0.0, b = 0.0;

	for (int ky = 0; ky < FILTER_SIZE; ky++) {
		for (int kx = 0; kx < FILTER_SIZE; kx++) {
			int i = index(clamp(x + kx - FILTER_HALFSIZE, src.xsize / 4), clamp(y + ky - FILTER_HALFSIZE, src.ysize), src);
			unsigned int pixel = ((int*)src.ptr)[i];
			const float k = gaussian[(ky * FILTER_SIZE) + kx];
			b += (float)((pixel & BLUE_MASK) >> 16) * k;
			g += (float)((pixel & GREEN_MASK) >> 8) * k;
			r += (float)((pixel & RED_MASK)) * k;
		}
	}
	unsigned int dpixel = 0x00000000
		| ((((int)b) << 16) & BLUE_MASK)
		| ((((int)g) << 8) & GREEN_MASK)
		| (((int)r) & RED_MASK);
	((int*)dst.ptr)[index(x, y, dst)] = dpixel;
}


void setupGaussian(float** d_gaussian) 
{
	float gaussian[FILTER_SIZE][FILTER_SIZE];
	double sigma = 5.0;
	double mean = FILTER_SIZE / 2;
	for (int x = 0; x < FILTER_SIZE; ++x)
	{
		for (int y = 0; y < FILTER_SIZE; ++y)
		{
			double g = exp(-0.5 * (pow((x - mean) / sigma, 2.0) + pow((y - mean) / sigma, 2.0))) / (2 * PI * sigma * sigma);
			gaussian[y][x] = (float)g;
		}
	}
	float sum = 0.0;
	for (int x = 0; x < FILTER_SIZE; ++x) 
	{
		for (int y = 0; y < FILTER_SIZE; ++y)
		{
			sum += gaussian[y][x];
		}
	}
	for (int x = 0; x < FILTER_SIZE; ++x) 
	{
		for (int y = 0; y < FILTER_SIZE; ++y)
		{
			gaussian[y][x] /= sum;
		}
	}

	cudaError_t cudaStatus = cudaMalloc(d_gaussian, FILTER_SIZE * FILTER_SIZE * sizeof(float));
	checkCudaErrors(cudaStatus);
	cudaStatus = cudaMemcpy(*d_gaussian, &gaussian[0], FILTER_SIZE * FILTER_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	checkCudaErrors(cudaStatus);
}

cudaEvent_t start, stop;

void startTimer() {
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
}

float stopTimer() {
	float time;
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	return time;
}

struct pixel_t {
	unsigned char __unused, red, green, blue;
};
void apply_blur(const char* INPUT_BMP_PATH,const char* OUTPUT_REFERENCE_BMP_PATH,const char* OUTPUT_BMP_PATH) {

	bitmap_image img(INPUT_BMP_PATH);

	if (!img) {
		printf("Error - Failed to open: %s \r\b", INPUT_BMP_PATH);
		return;
	}

	if (img.height() % BLOCKDIM != 0) {
		printf("ERROR: image height (%d) must be a multiple of the block size (%d)\n", img.height(), BLOCKDIM);
		return;
	}
	if (img.width() % BLOCKDIM != 0) {
		printf("ERROR: image width (%d) must be a multiple of the block size (%d)\n", img.width(), BLOCKDIM);
		return;
	}

	const int IMG_WIDTH_BYTES = img.width() * 4;

	pixel_t* h_buf = new pixel_t[img.width() * img.height()];

	for (unsigned int y = 0; y < img.height(); y++)
	{
		for (unsigned int x = 0; x < img.width(); x++)
		{
			pixel_t* pixel = &h_buf[(y * img.width()) + x];
			img.get_pixel(x, y, pixel->red, pixel->green, pixel->blue);
		}
	}

	cudaError_t cudaStatus;
	cudaStatus = cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	checkCudaErrors(cudaStatus);
	int kernelTimeout;
	cudaStatus = cudaDeviceGetAttribute(&kernelTimeout, cudaDevAttrKernelExecTimeout, 0/*device*/);
	checkCudaErrors(cudaStatus);
	if (kernelTimeout != 0) {
		printf("WARNING: kernel timeout is enabled! %d \r\n", kernelTimeout);
	}

	// COPY IMAGE BUFFERS AND FILTER TO DEVICE
	startTimer();
	// Returns a cudaExtent based on the specified input parameters (width_in_bytes,height_elements,depth_in_elements)
	cudaExtent extent = make_cudaExtent(IMG_WIDTH_BYTES, img.height(), 1);
	//printf("Extent dimensions are [%zd,%zd,%zd]\n", extent.width,extent.height,extent.depth);

	cudaPitchedPtr d_src, d_dst;
	// -- Allocate a pitched memory array d_src;
	// Allocates at least width * height * depth bytes of linear memory on the device and returns a 
	// cudaPitchedPtr in which ptr is a pointer to the allocated memory. The function may pad the 
	// allocation to ensure hardware alignment requirements are met. 
	// The pitch returned in the pitch field of pitchedDevPtr is the width in bytes of the allocation.
	// Pitched device pointer has pitch, ptr, x_size, y_size variables
	cudaStatus = cudaMalloc3D(&d_src, extent);
	checkCudaErrors(cudaStatus);

	// -- Copy from buffer to pitched memory array d_src;
	// Copies a matrix (height rows of width bytes each) from the memory area pointed to by src 
	// to the memory area pointed to by dst, where kind is one of cudaMemcpyHostToHost, 
	// cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, or cudaMemcpyDeviceToDevice, and 
	// specifies the direction of the copy. dpitch and spitch are the widths in memory in bytes 
	// of the 2D arrays pointed to by dst and src, including any padding added to the end of each row.
	cudaStatus = cudaMemcpy2D(d_src.ptr, d_src.pitch,
		h_buf, IMG_WIDTH_BYTES, IMG_WIDTH_BYTES, img.height(),
		cudaMemcpyHostToDevice);
	checkCudaErrors(cudaStatus);

	// -- Allocate a pitched memory array d_dst;
	cudaStatus = cudaMalloc3D(&d_dst, extent);
	checkCudaErrors(cudaStatus);

	// -- Fill pitched memory with zeros?
	// Sets to the specified value value a matrix (height rows of width bytes each) pointed
	// to by dstPtr. pitch is the width in bytes of the 2D array pointed to by dstPtr, 
	// including any padding added to the end of each row.
	cudaStatus = cudaMemset2D(d_dst.ptr, d_dst.pitch, 0, IMG_WIDTH_BYTES, img.height());
	checkCudaErrors(cudaStatus);

	// Create a gaussian kernel with window size 15, variance 5 and mean value 2.5
	float* d_gaussian;
	setupGaussian(&d_gaussian);

	printf("Copy to device:  %3.1f ms \n", stopTimer());

	// LAUNCH KERNEL

	//printf("\nBlock dimension is %d.\n", BLOCKDIM);
	//printf("The number of blocks in grid is [%d,%d].\n", (img.width() / BLOCKDIM), (img.height() / BLOCKDIM));
	//printf("The number of threads per block is [%d,%d].\n", BLOCKDIM, BLOCKDIM);

	for (int i = 0; i < 1; i++) 
	{
		startTimer();

		dim3 blocksInGrid(img.width() / BLOCKDIM, img.height() / BLOCKDIM);
		dim3 threadsPerBlock(BLOCKDIM, BLOCKDIM);

		// Launch kernel with performance bug fix
		blurGlobal << <blocksInGrid, threadsPerBlock >> > (d_src, d_dst, d_gaussian);

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		checkCudaErrors(cudaStatus);

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		checkCudaErrors(cudaStatus);

		printf("Kernel time:  %3.1f ms \n", stopTimer());
	}

	// COPY  OUTPUT IMAGE BACK TO HOST
	startTimer();
	cudaStatus = cudaMemcpy2D(h_buf, IMG_WIDTH_BYTES,
		d_dst.ptr, d_dst.pitch, IMG_WIDTH_BYTES, d_dst.ysize,
		cudaMemcpyDeviceToHost);
	checkCudaErrors(cudaStatus);

	printf("Copy from device:  %3.1f ms \n", stopTimer());

	// WRITE OUT UPDATED IMAGE
	for (unsigned int y = 0; y < img.height(); y++)
	{
		for (unsigned int x = 0; x < img.width(); x++) 
		{
			const pixel_t* p = &h_buf[(y * img.width()) + x];
			img.set_pixel(x, y, p->red, p->green, p->blue);
		}
	}
	img.save_image(OUTPUT_BMP_PATH);

	// VALIDATION

	bool validated = true;
	bitmap_image ref(OUTPUT_REFERENCE_BMP_PATH);

	if (img.height() != ref.height()) 
	{
		fprintf(stderr, "Image height should be %u but was %u \r\n", ref.height(), img.height());
		validated = false;
	}
	if (img.width() != ref.width()) 
	{
		fprintf(stderr, "Image width should be %u but was %u \r\n", ref.width(), img.width());
		validated = false;
	}
	unsigned int differingPixels = 0;
	double squareDiffSum = 0;
	for (unsigned int y = 0; y < ref.height(); y++) 
	{
		for (unsigned int x = 0; x < ref.width(); x++)
		{
			rgb_t refPixel, imgPixel;
			ref.get_pixel(x, y, refPixel);
			img.get_pixel(x, y, imgPixel);
			if (refPixel.red != imgPixel.red ||
				refPixel.green != imgPixel.green ||
				refPixel.blue != imgPixel.blue) {
				differingPixels++;
				
				// compute square difference
				unsigned int redDiff = refPixel.red - imgPixel.red;
				unsigned int greenDiff = refPixel.green - imgPixel.green;
				unsigned int blueDiff = refPixel.blue - imgPixel.blue;
				squareDiffSum += (redDiff * redDiff) + (greenDiff * greenDiff) + (blueDiff * blueDiff);
			}
		}
	}
	if (0 != differingPixels)
	{
		fprintf(stderr, "Found %u pixels that differ from the reference image \r\n", differingPixels);
		double rmsd = sqrt(squareDiffSum / (ref.height() * ref.width()));
		fprintf(stderr, "RMSD of pixel rgb values is %3.5f \r\n", rmsd);
		validated = false;
	}

	if (validated) 
	{
		printf("Validation passed :-) \r\n");
	}


	// CLEANUP

	cudaStatus = cudaFree(d_src.ptr);
	checkCudaErrors(cudaStatus);
	cudaStatus = cudaFree(d_dst.ptr);
	checkCudaErrors(cudaStatus);
	cudaStatus = cudaFree(d_gaussian);
	checkCudaErrors(cudaStatus);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	checkCudaErrors(cudaStatus);

	delete[] h_buf;

	return;
}

