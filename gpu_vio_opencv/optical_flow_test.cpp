#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <sys/time.h>

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

#define trials_count 100

struct timeval begin_ff, end_ff, begin_tt, end_tt, begin_tf, end_tf;
struct timeval begin_pdet, end_pdet, begin_pddh, end_pddh, begin_pdhd, end_pdhd;
struct timeval begin_dop, end_dop;
struct timeval begin_gfop, end_gfop;
struct timeval begin_gfd, end_gfd;
struct timeval begin_stc_1, end_stc_1;
struct timeval begin_stc_2, end_stc_2;
struct timeval begin_stc_3, end_stc_3;
struct timeval begin_ffop, end_ffop;

std::string type2str(int type) 
{
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}


void Mat_to_vector_KeyPoint(cv::Mat& mat, std::vector<cv::KeyPoint>& v_kp)
{
    v_kp.clear();

    for(int i=0; i<mat.rows; i++)
    {
        cv::Vec<float, 7> v = mat.at< cv::Vec<float, 7> >(i,0);
        cv::KeyPoint kp(v[0], v[1], v[2], v[3], v[4], (int)v[5], (int)v[6]);
        v_kp.push_back(kp);
    }
    return;
}


int main(int arg, char const *argv[])
{

	// Find if a CUDA::GPU enabled device exists
	int gpu_count = cv::gpu::getCudaEnabledDeviceCount();

	if (gpu_count < 1) {
		printf("CUDA Enabled device does not exist.\n");
		return 0;
	}

	// Set device to first CUDA::GPU enabled device in chain
	cv::gpu::setDevice(0);

	std::cout << "---------------------------------------------------------------\n";
	std::cout << "------------------ FAST FEATURE DETECTION ---------------------\n";
	std::cout << "---------------------------------------------------------------\n";

	// Create host side placeholders for input image
	cv::Mat h_ip_img_l;//h_ip_img_r;
	cv::Mat h_op_img_l;//h_op_img_r;

	//cv::Mat h_points;
	std::vector<cv::KeyPoint> h_points_left;
	//std::vector<cv::KeyPoint> h_points_right;

	// Read input image
	h_ip_img_l = cv::imread("/home/nvidia/gpu-vio/left00001.png");
	//h_ip_img_r = cv::imread("/home/nvidia/gpu-vio/right00001.png");

	cv::cvtColor(h_ip_img_l, h_ip_img_l, CV_RGB2GRAY);
	//cv::cvtColor(h_ip_img_r, h_ip_img_r, CV_RGB2GRAY);


	// --------------- TESTING FAST FEATURE DETECTOR ---------------

	// Create fast feature detection object
	cv::gpu::FAST_GPU fast_gpu_obj(20,true);

	// Create device side placeholders for input image
	cv::gpu::GpuMat d_left;//d_right;
	cv::gpu::GpuMat points_left;//points_right;

	// ----------- first pass --------------
	// Upload input image from host to device
	d_left.upload(h_ip_img_l);
	//d_right.upload(h_ip_img_r);

	// Run FAST FEATURE DETECTION on device side image
	// -- run once independently to avoid counting init overhead
	fast_gpu_obj(d_left,cv::gpu::GpuMat(),points_left);
	//fast_gpu_obj(d_right,cv::gpu::GpuMat(),points_right);
	// -------------------------------------
	
	double t_h2d_avg = 0;

	gettimeofday(&begin_ff,NULL);
        for (int i = 0; i < trials_count; i++)
        {
	    gettimeofday(&begin_tt,NULL);
            d_left.upload(h_ip_img_l);
	    //d_right.upload(h_ip_img_r);
	    gettimeofday(&end_tt,NULL);
            const double eptt = end_tt.tv_sec + end_tt.tv_usec / 1e6 - begin_tt.tv_sec - begin_tt.tv_usec / 1e6;
	    t_h2d_avg = t_h2d_avg + eptt;
            fast_gpu_obj(d_left, cv::gpu::GpuMat(), points_left);
	    //fast_gpu_obj(d_right, cv::gpu::GpuMat(), points_right);
        }
	gettimeofday(&end_ff,NULL);

	t_h2d_avg = t_h2d_avg / trials_count;
	std::cout << "Transfer to Device (Overhead) : " << (t_h2d_avg*1000.0)<< " ms" << std::endl << std::endl;

	gettimeofday(&begin_tf,NULL);
	fast_gpu_obj.downloadKeypoints(points_left,h_points_left);
	//fast_gpu_obj.downloadKeypoints(points_right,h_points_right);
	// Download image from device to host
	d_left.download(h_op_img_l);
	//d_right.download(h_op_img_r);
	gettimeofday(&end_tf,NULL);

        const double eptf = end_tf.tv_sec + end_tf.tv_usec / 1e6 - begin_tf.tv_sec - begin_tf.tv_usec / 1e6;
        std::cout << "Transfer from Device (Overhead) : " << (eptf*1000.0)<< " ms" << std::endl << std::endl;


	//cv::Mat img_keypoints_1;
	//cv::Mat img_keypoints_2;
	//cv::drawKeypoints(h_op_img_l, h_points_left, img_keypoints_1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );
	//cv::drawKeypoints(h_op_img_r, h_points_right, img_keypoints_2, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );

	
	// Display output image on device
	//cv::imshow("Test Image:L",img_keypoints_1);
	//cv::imshow("Test Image:R",img_keypoints_2);
	//cv::waitKey(0);
	

	//cv::imwrite("test_fast_feature_20_l.png",img_keypoints_1);
	//cv::imwrite("test_fast_feature_20_r.png",img_keypoints_2);

        const double ep = end_ff.tv_sec + end_ff.tv_usec / 1e6 - begin_ff.tv_sec - begin_ff.tv_usec / 1e6;
        std::cout << "Time Elapsed for Fast Feature Computation : " << (ep*1000.0)/trials_count << " ms" << std::endl << std::endl;




	// --------------- TESTING PYRAMID DOWNSAMPLING ---------------
	std::cout << "---------------------------------------------------------------\n";
	std::cout << "------------------ PYRAMID DOWNSAMPLING -----------------------\n";
	std::cout << "---------------------------------------------------------------\n";

	cv::Mat h_ip_img_l_t1 = cv::imread("/home/nvidia/gpu-vio/right00001.png");
	cv::cvtColor(h_ip_img_l_t1, h_ip_img_l_t1, CV_RGB2GRAY);

	cv::gpu::GpuMat d_t;
	cv::gpu::GpuMat d_t1;

	cv::gpu::GpuMat d_t_s1,d_t_s2,d_t_s3;
	cv::gpu::GpuMat d_t1_s1,d_t1_s2,d_t1_s3;

	cv::Mat h_s1_t;
	cv::Mat h_s2_t;
	cv::Mat h_s3_t;

	d_t.upload(h_ip_img_l);
	d_t1.upload(h_ip_img_l_t1);

	cv::gpu::pyrDown(d_t,d_t_s1);
	cv::gpu::pyrDown(d_t_s1,d_t_s2);
	cv::gpu::pyrDown(d_t_s2,d_t_s3);

	cv::gpu::pyrDown(d_t1,d_t1_s1);
	cv::gpu::pyrDown(d_t1_s1,d_t1_s2);
	cv::gpu::pyrDown(d_t1_s2,d_t1_s3);

	
	double pdhd_avg = 0;
	gettimeofday(&begin_pdet,NULL); // start main timer for PyrDown
        for (int i = 0; i < trials_count; i++)
        {
	    gettimeofday(&begin_pdhd,NULL); // start timer for IMAGE UPLOAD to GPU
            d_t.upload(h_ip_img_l);
	    gettimeofday(&end_pdhd,NULL); // end timer for IMAGE UPLOAD to GPU

            const double ephd = end_pdhd.tv_sec + end_pdhd.tv_usec / 1e6 - begin_pdhd.tv_sec - begin_pdhd.tv_usec / 1e6;
	    pdhd_avg = pdhd_avg + ephd;

	    // Pyramid Downsampling 3 Levels
	    cv::gpu::pyrDown(d_t,d_t_s1);
	    cv::gpu::pyrDown(d_t_s1,d_t_s2);
	    cv::gpu::pyrDown(d_t_s2,d_t_s3);
        }
	gettimeofday(&end_pdet,NULL); // end main timer for PyrDown
	std::cout << "Transfer to Device (Overhead) : " << (pdhd_avg*1000.0)/trials_count << " ms" << std::endl << std::endl;

	gettimeofday(&begin_pddh,NULL); // start timer for IMAGE DOWNLOAD from GPU
	// Download image from device to host
	d_t_s1.download(h_s1_t);
	d_t_s2.download(h_s2_t);
	d_t_s3.download(h_s3_t);
	gettimeofday(&end_pddh,NULL); // end timer for IMAGE DOWNLOAD from GPU

        const double epdh = end_pddh.tv_sec + end_pddh.tv_usec / 1e6 - begin_pddh.tv_sec - begin_pddh.tv_usec / 1e6;
        std::cout << "Transfer from Device (Overhead) : " << (epdh*1000.0)<< " ms" << std::endl << std::endl;

	const double ep_pd = end_pdet.tv_sec + end_pdet.tv_usec / 1e6 - begin_pdet.tv_sec - begin_pdet.tv_usec / 1e6;
        std::cout << "Time Elapsed for 3 Level Pyramid Downsampling : " << (ep_pd*1000.0)/trials_count << " ms" << std::endl << std::endl;

	/*
	cv::imshow("Pyramid Downsampled - 3",h_s3_t);
	cv::waitKey(0);
	cv::imshow("Pyramid Downsampled - 2",h_s2_t);
	cv::waitKey(0);
	cv::imshow("Pyramid Downsampled - 1",h_s1_t);
	cv::waitKey(0);
	*/	


	std::cout << "---------------------------------------------------------------\n";
	std::cout << "----------------- OPTICAL FLOW ESTIMATION ---------------------\n";
	std::cout << "--------------------- ( fast feature ) ------------------------\n";

	int opflow_windowsize = 31;
	int opflow_levels = 3;
	int opflow_iters = 30;

	cv::gpu::PyrLKOpticalFlow opflow_object;
	opflow_object.winSize.width = opflow_windowsize;
	opflow_object.winSize.height = opflow_windowsize;
	opflow_object.maxLevel = opflow_levels;
	opflow_object.iters = opflow_iters;

	cv::gpu::FAST_GPU fast_gpu_opf(20,true);

	cv::Mat h_opf_t0 = cv::imread("/home/nvidia/gpu-vio/data/left00035.png");
	cv::Mat h_opf_t1 = cv::imread("/home/nvidia/gpu-vio/data/left00036.png");

	cv::cvtColor(h_opf_t0, h_opf_t0, CV_RGB2GRAY);
	cv::cvtColor(h_opf_t1, h_opf_t1, CV_RGB2GRAY);

	cv::gpu::GpuMat d_opf_t0; // device-side holder for img t=0
	cv::gpu::GpuMat d_opf_t1; // device-side holder for img t=1

	cv::gpu::GpuMat d_status_mat; // create device-side status variable

	d_opf_t0.upload(h_opf_t0); // copy img t=0 from host to device
	d_opf_t1.upload(h_opf_t1); // copy img t=1 from host to device

	cv::gpu::GpuMat d_points_t0; // device-side holder for keypoint features for img t=0
	cv::gpu::GpuMat d_points_t1; // device-side holder for keypoint features for img t=1

        fast_gpu_opf(d_opf_t0, cv::gpu::GpuMat(), d_points_t0); // calculate fast features on img t=0 on the device


	std::vector<cv::KeyPoint> fin_ff_vec0;
	fast_gpu_opf.downloadKeypoints(d_points_t0,fin_ff_vec0);

	// ---- stupid type conversion starts here
	gettimeofday(&begin_stc_1,NULL);
	std::vector<cv::Point2f> points;
	std::vector<cv::KeyPoint>::iterator it;

	for( it= fin_ff_vec0.begin(); it!= fin_ff_vec0.end();it++)
	{
	    points.push_back(it->pt);
	}
	cv::Mat h_points_t0(points);

	//std::cout << h_points_t0.rows << " " << h_points_t0.cols << std::endl;

	std::vector<cv::KeyPoint> fin2_ff_vec0;
	Mat_to_vector_KeyPoint(h_points_t0,fin2_ff_vec0);
	gettimeofday(&end_stc_1,NULL);
	// ---- stupid type conversion ends here

	cv::Mat img_keypointsff_t0;
	cv::drawKeypoints(h_opf_t0, fin2_ff_vec0, img_keypointsff_t0, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );
	cv::imshow("T0 Keypoints: FF",img_keypointsff_t0);
	cv::waitKey(0);

	// ---- stupid type conversion continues here
	gettimeofday(&begin_stc_2,NULL);
	cv::gpu::GpuMat d_points_t0_fix;
	cv::transpose(h_points_t0,h_points_t0);
	d_points_t0_fix.upload(h_points_t0); 	// ---- stupid type conversion requires re-upload

	std::string ty =  type2str( h_points_t0.type() );
	printf("T=0 Feature Matrix: %s %dx%d \n", ty.c_str(), h_points_t0.cols, h_points_t0.rows );
	gettimeofday(&end_stc_2,NULL);
	// ---- stupid type conversion continuation ends here
	

	gettimeofday(&begin_ffop,NULL); // start main timer for Optical Flow
        for (int i = 0; i < trials_count; i++)
        {
		opflow_object.sparse(d_opf_t0, d_opf_t1, d_points_t0_fix, d_points_t1, d_status_mat);
        }
	gettimeofday(&end_ffop,NULL); // end main timer for Optical Flow

	const double ep_ffof = end_ffop.tv_sec + end_ffop.tv_usec / 1e6 - begin_ffop.tv_sec - begin_ffop.tv_usec / 1e6;
        std::cout << "Time Elapsed for Optical Flow Estimation (FF) : " << (ep_ffof*1000.0)/trials_count << " ms" << std::endl << std::endl;

	// ---- more stupid type conversion
	gettimeofday(&begin_stc_3,NULL);
	cv::Mat h_points_t1;
	d_points_t1.download(h_points_t1);

	std::string ty2 =  type2str( h_points_t1.type() );
	printf("T=1 Feature Matrix: %s %dx%d \n", ty2.c_str(), h_points_t1.cols, h_points_t1.rows );

	cv::transpose(h_points_t1,h_points_t1);
	std::vector<cv::KeyPoint> fin2_ff_vec1;
	Mat_to_vector_KeyPoint(h_points_t1,fin2_ff_vec1);
	gettimeofday(&end_stc_3,NULL);
	// ---- stupid type conversion finally ends

	cv::Mat img_keypointsff_t1;
	cv::drawKeypoints(h_opf_t1, fin2_ff_vec1, img_keypointsff_t1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );
	cv::imshow("T1 Keypoints: FF",img_keypointsff_t1);
	cv::waitKey(0);

	cv::imwrite("optical_flow_t0_ff.png",img_keypointsff_t0);
	cv::imwrite("optical_flow_t1_ff.png",img_keypointsff_t1);


	const double ep_stc1 = end_stc_1.tv_sec + end_stc_1.tv_usec / 1e6 - begin_stc_1.tv_sec - begin_stc_1.tv_usec / 1e6;
        std::cout << "Stupid Type Conversion Part 1 : " << (ep_stc1*1000.0) << " ms" << std::endl << std::endl;

	const double ep_stc2 = end_stc_2.tv_sec + end_stc_2.tv_usec / 1e6 - begin_stc_2.tv_sec - begin_stc_2.tv_usec / 1e6;
        std::cout << "Stupid Type Conversion Part 2 : " << (ep_stc2*1000.0) << " ms" << std::endl << std::endl;

	const double ep_stc3 = end_stc_3.tv_sec + end_stc_3.tv_usec / 1e6 - begin_stc_3.tv_sec - begin_stc_3.tv_usec / 1e6;
        std::cout << "Stupid Type Conversion Part 3 : " << (ep_stc3*1000.0) << " ms" << std::endl << std::endl;


	// --------------------- DENSE OPTICAL FLOW ESTIMATION ---------------------------
	/*
	std::cout << std::endl;
	std::cout << "---------------------------------------------------------------\n";
	std::cout << "----------------- OPTICAL FLOW ESTIMATION ---------------------\n";
	std::cout << "-------------------- ( dense optflow ) ------------------------\n";
	cv::gpu::GpuMat d_h_comp;
	cv::gpu::GpuMat d_v_comp;

	gettimeofday(&begin_dop,NULL); // start main timer for dense Optical Flow
        for (int i = 0; i < trials_count; i++)
        {
		opflow_object.dense(d_opf_t0,d_opf_t1,d_h_comp,d_v_comp);
        }
	gettimeofday(&end_dop,NULL); // end main timer for dense Optical Flow

	const double ep_dof = end_dop.tv_sec + end_dop.tv_usec / 1e6 - begin_dop.tv_sec - begin_dop.tv_usec / 1e6;
        std::cout << "Time Elapsed for Dense Optical Flow Estimation : " << (ep_dof*1000.0)/trials_count << " ms" << std::endl << std::endl;
	*/


	// ----------------------- GOOD FEATURES TO TRACK TEST ---------------------------
	std::cout << std::endl;
	std::cout << "---------------------------------------------------------------\n";
	std::cout << "----------------- OPTICAL FLOW ESTIMATION ---------------------\n";
	std::cout << "-------------------- ( good features ) ------------------------\n";
	cv::gpu::GoodFeaturesToTrackDetector_GPU gftt;

	cv::gpu::GpuMat d_good_f_t0;
	gftt(d_opf_t0,d_good_f_t0); // detect good features for img t=0

	cv::Mat h_good_f_t0;
	d_good_f_t0.download(h_good_f_t0); // download to host

	cv::transpose(h_good_f_t0,h_good_f_t0); // transpose such that each row is a feature
	std::vector<cv::KeyPoint> fin_vec0;
	Mat_to_vector_KeyPoint(h_good_f_t0,fin_vec0);

	cv::Mat img_keypointsf0;
	cv::drawKeypoints(h_opf_t0, fin_vec0, img_keypointsf0, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );
	cv::imshow("T0 Keypoints",img_keypointsf0);
	cv::waitKey(0);
	
	cv::gpu::GpuMat d_good_f_t1;
	cv::Mat h_good_f_t1;

	gettimeofday(&begin_gfop,NULL); // start main timer for Optical Flow
        for (int i = 0; i < trials_count; i++)
        {
		opflow_object.sparse(d_opf_t0, d_opf_t1, d_good_f_t0, d_good_f_t1, d_status_mat);
        }
	gettimeofday(&end_gfop,NULL); // end main timer for Optical Flow

	const double ep_gfof = end_gfop.tv_sec + end_gfop.tv_usec / 1e6 - begin_gfop.tv_sec - begin_gfop.tv_usec / 1e6;
        std::cout << "Time Elapsed for Optical Flow Estimation (GF) : " << (ep_gfof*1000.0)/trials_count << " ms" << std::endl << std::endl;

	gettimeofday(&begin_gfd,NULL); 
	d_good_f_t1.download(h_good_f_t1);
	gettimeofday(&end_gfd,NULL); 

	const double ep_gfd = end_gfd.tv_sec + end_gfd.tv_usec / 1e6 - begin_gfd.tv_sec - begin_gfd.tv_usec / 1e6;
        std::cout << "Time Elapsed for Device to Host Transfer : " << (ep_gfd*1000.0) << " ms" << std::endl << std::endl;

	cv::transpose(h_good_f_t1,h_good_f_t1); // transpose such that each row is a feature
	std::vector<cv::KeyPoint> fin_vec1;
	Mat_to_vector_KeyPoint(h_good_f_t1,fin_vec1);

	cv::Mat img_keypointsf1;
	cv::drawKeypoints(h_opf_t1, fin_vec1, img_keypointsf1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );
	cv::imshow("T1 Keypoints",img_keypointsf1);
	cv::waitKey(0);

	cv::imwrite("optical_flow_t0.png",img_keypointsf0);
	cv::imwrite("optical_flow_t1.png",img_keypointsf1);

	//gftt_f_h_t1 = gftt_f_h_t1.reshape(1,gftt_f_h_t1.rows*2);

}


