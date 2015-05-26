#pragma once

#ifndef __LBFTYPES_H
#define __LBFTYPES_H
#endif

//openCV library
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

//LBF type definitions
#define SIZE_DOUBLE		//define LBF Type

#if defined SIZE_FLOAT
typedef float			LBF_DATA;
typedef cv::Point2f		LBF_POINT;
#define LBF_MAT_TYPE	CV_32FC1

#elif defined SIZE_DOUBLE
typedef	double			LBF_DATA;
typedef cv::Point2d		LBF_POINT;
#define LBF_MAT_TYPE	CV_64FC1

#endif