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

extern int stereo_match(const char *lfname, const char *rfname, unsigned int height, unsigned int width);

int main()
{
    const char* lfname = "data/stereo.im0.640x533.ppm";
    const char* rfname = "data/stereo.im1.640x533.ppm";

    //const char* lfname = "data/im0.ppm";
    //const char* rfname = "data/im1.ppm";
	
    Mat img_left = imread(lfname);
    Mat img_right = imread(rfname);
    Size img_size = img_left.size();
    unsigned int height = img_size.height;
    unsigned int width = img_size.width;

    stereo_match(lfname, rfname, height, width);

    return 0;
}
