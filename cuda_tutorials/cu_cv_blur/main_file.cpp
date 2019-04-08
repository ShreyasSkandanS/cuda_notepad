#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <stdio.h>
#include <iostream>
#include <ctime>
#include <sys/time.h>

using namespace std;
using namespace cv;

extern void apply_blur(const char *a, const char *b, const char *c);

int main()
{
    const char* INPUT_IMG = "data/steel_wool_small.bmp";
    const char* OUTPUT_IMG = "data/steel_wool_output.bmp";
    const char* REF_IMG = "data/steel_wool_small_reference_output.bmp";
	
    Mat imgi = imread(INPUT_IMG);
    imshow("Input",imgi);
    waitKey();

    apply_blur(INPUT_IMG,REF_IMG,OUTPUT_IMG);

    Mat imgo = imread(OUTPUT_IMG);
    imshow("Output",imgo);
    waitKey();

    return 0;
}
