/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* A CUDA program that demonstrates how to compute a stereo disparity map using
 *   SIMD SAD (Sum of Absolute Difference) intrinsics
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, kernels
#include <cuda_runtime.h>
#include "stereoDisparity_kernel.cuh"

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

// includes, project
#include <helper_functions.h>  // helper for shared that are common to CUDA Samples
#include <helper_cuda.h>       // helper for checking cuda initialization and error checking
#include <helper_string.h>     // helper functions for string parsing


//static const char *sSDKsample = "[stereoDisparity]\0";

int iDivUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}


////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(const char *lfname, const char *rfname, unsigned int h, unsigned int w);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
stereo_match(const char *lfname, const char *rfname, unsigned int h, unsigned int w)
{
    runTest(lfname, rfname, h, w);
}

////////////////////////////////////////////////////////////////////////////////
//! CUDA Sample for calculating depth maps
////////////////////////////////////////////////////////////////////////////////
void
runTest(const char *fname0, const char *fname1, unsigned int h, unsigned int w)
{
    cv::Mat l_img = cv::imread(fname0);
    cv::Mat r_img = cv::imread(fname1);
    cv::imshow("Left",l_img);
    cv::waitKey();

    uint32_t cols, rows, size;
    cols = l_img.cols;
    rows = r_img.rows;
    size = cols*rows*3;

    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, 0));
    printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
           deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

    StopWatchInterface *timer;
    sdkCreateTimer(&timer);

    // Search parameters
    int minDisp = -16;
    int maxDisp = 112;

    // Load image data
    //allocate mem for the images on host side
    //initialize pointers to NULL to request lib call to allocate as needed
    // PPM images are loaded into 4 byte/pixel memory (RGBX)
    unsigned char *h_img0 = NULL;
    unsigned char *h_img1 = NULL;
    unsigned int temp1, temp2;

    printf("Loaded <%s> as image 0\n", fname0);
    if (!sdkLoadPPM4ub(fname0, &h_img0, &temp1, &temp2))
        fprintf(stderr, "Failed to load <%s>\n", fname0);
    printf("Loaded <%s> as image 1\n", fname1);
    if (!sdkLoadPPM4ub(fname1, &h_img1, &temp1, &temp2))
        fprintf(stderr, "Failed to load <%s>\n", fname1);
    printf("w is %d | ",w);
    printf("h is %d\n",h);

    dim3 numThreads = dim3(blockSize_x, blockSize_y, 1);
    dim3 numBlocks = dim3(iDivUp(w, numThreads.x), iDivUp(h, numThreads.y));

    unsigned int numData = (unsigned int)(rows*cols);
    printf("numData is: %d\n",numData);
    unsigned int memSize = (unsigned int)(rows*cols*sizeof(int));
    printf("Memsize is: %d\n",memSize);

    //allocate mem for the result on host side
    unsigned int *h_odata = (unsigned int *)malloc(memSize);

    //initialize the memory
    for (unsigned int i = 0; i < numData; i++)
        h_odata[i] = 0;

    // allocate device memory for result
    unsigned int *d_odata, *d_img0, *d_img1;

    checkCudaErrors(cudaMalloc((void **) &d_odata, memSize));
    checkCudaErrors(cudaMalloc((void **) &d_img0, memSize));
    checkCudaErrors(cudaMalloc((void **) &d_img1, memSize));

    // copy host memory to device to initialize to zeros
    checkCudaErrors(cudaMemcpy(d_img0,  h_img0, memSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_img1,  h_img1, memSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_odata, h_odata, memSize, cudaMemcpyHostToDevice));

    // test opencv
    uint8_t *d_im1;
    checkCudaErrors(cudaMalloc((void **) &d_im1, sizeof(uint8_t)*size));
    checkCudaErrors(cudaMemcpy(d_im1,  l_img.ptr<uint8_t>(), sizeof(uint8_t)*size, cudaMemcpyHostToDevice));
    uint8_t *d_im2;
    checkCudaErrors(cudaMalloc((void **) &d_im2, sizeof(uint8_t)*size));
    checkCudaErrors(cudaMemcpy(d_im2,  r_img.ptr<uint8_t>(), sizeof(uint8_t)*size, cudaMemcpyHostToDevice));
    printf("Passed\n");

    size_t offset = 0;
    cudaChannelFormatDesc ca_desc0 = cudaCreateChannelDesc<unsigned int>();
    cudaChannelFormatDesc ca_desc1 = cudaCreateChannelDesc<unsigned int>();

    tex2Dleft.addressMode[0] = cudaAddressModeClamp;
    tex2Dleft.addressMode[1] = cudaAddressModeClamp;
    tex2Dleft.filterMode     = cudaFilterModePoint;
    tex2Dleft.normalized     = false;
    tex2Dright.addressMode[0] = cudaAddressModeClamp;
    tex2Dright.addressMode[1] = cudaAddressModeClamp;
    tex2Dright.filterMode     = cudaFilterModePoint;
    tex2Dright.normalized     = false;
    checkCudaErrors(cudaBindTexture2D(&offset, tex2Dleft,  d_img0, ca_desc0, w, h, w*4));
    assert(offset == 0);

    checkCudaErrors(cudaBindTexture2D(&offset, tex2Dright, d_img1, ca_desc1, w, h, w*4));
    assert(offset == 0);

    // First run the warmup kernel (which we'll use to get the GPU in the correct max power state
    stereoDisparityKernel<<<numBlocks, numThreads>>>(d_img0, d_img1, d_odata, w, h, minDisp, maxDisp);
    //stereoDisparityKernel<<<numBlocks, numThreads>>>((unsigned int*)d_im1, (unsigned int*)d_im2, d_odata, w, h, minDisp, maxDisp);

    cudaDeviceSynchronize();

    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    printf("Launching CUDA stereoDisparityKernel()\n");

    // Record the start event
    checkCudaErrors(cudaEventRecord(start, NULL));

    // launch the stereoDisparity kernel
    stereoDisparityKernel<<<numBlocks, numThreads>>>(d_img0, d_img1, d_odata, w, h, minDisp, maxDisp);
    //stereoDisparityKernel<<<numBlocks, numThreads>>>((unsigned int*)d_im1, (unsigned int*)d_im2, d_odata, w, h, minDisp, maxDisp);
    
    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, NULL));

    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));

    // Check to make sure the kernel didn't fail
    getLastCudaError("Kernel execution failed");

    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    //Copy result from device to host for verification
    checkCudaErrors(cudaMemcpy(h_odata, d_odata, memSize, cudaMemcpyDeviceToHost));

    printf("Input Size  [%dx%d], ", w, h);
    printf("Kernel size [%dx%d], ", (2*RAD+1), (2*RAD+1));
    printf("Disparities [%d:%d]\n", minDisp, maxDisp);

    printf("GPU processing time : %.4f (ms)\n", msecTotal);
    printf("Pixel throughput    : %.3f Mpixels/sec\n", ((float)(w *h*1000.f)/msecTotal)/1000000);

    // calculate sum of resultant GPU image
    unsigned int checkSum = 0;

    for (unsigned int i=0 ; i<w *h ; i++)
    {
        checkSum += h_odata[i];
    }

    printf("GPU Checksum = %u, ", checkSum);

    // write out the resulting disparity image.
    unsigned char *dispOut = (unsigned char *)malloc(numData);
    int mult = 20;
    const char *fnameOut = "output_GPU.pgm";

    for (unsigned int i=0; i<numData; i++)
    {
        dispOut[i] = (int)h_odata[i]*mult;
    }

    printf("GPU image: <%s>\n", fnameOut);
    sdkSavePGM(fnameOut, dispOut, w, h);

    // cleanup memory
    checkCudaErrors(cudaFree(d_odata));
    checkCudaErrors(cudaFree(d_img0));
    checkCudaErrors(cudaFree(d_img1));

    if (h_odata != NULL) free(h_odata);
    if (h_img0 != NULL) free(h_img0);
    if (h_img1 != NULL) free(h_img1);
    if (dispOut != NULL) free(dispOut);

    sdkDeleteTimer(&timer);
}
