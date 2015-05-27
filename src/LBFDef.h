#pragma once

#ifndef __LBFDEF_H
#define __LBFDEF_H

#ifndef __SHAPE_H
#include "Shape.h"
#endif

#ifndef __LBFTYPES_H
#include "LBFTypes.h"
#endif

//openCV library
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#define nullptr 0

//LBF structs
typedef struct _Params
{
	Shape meanShape;

	// LBF common parameters
	int numLandmark;

	// initialize parameters for boosting gradient regression
	int numStage;

	// initialize parameters for shape data augmentation
	int augnumber;
	int augnumShift;
	int augnumRotate;
	int augnumScale;
	int flipflag; 
	int validPointsIdx[68];		// COFW? -> [1:17, 28, 31, 32, 34, 36, 37, 40, 43, 46, 49, 52, 55, 58];

	// initialize parameters for training random forest
	int binaryLength;

	int maxNumTrees;
	LBF_DATA numTreeDataRatio;	// range : (0, 1)
	std::vector<int> maxNumFeatures;	// { 1000, 1000, 1000, 500, 500, 500, 400, 400, ... }
	std::vector<LBF_DATA> maxRatioRadius;	// { 0.4, 0.3, 0.2, 0.15, 0.12, 0.10, 0.08, 0.06, 0.06, 0.05, ... }
	std::vector<LBF_DATA> radius;
	std::vector<LBF_DATA> angles;

	_Params()
		: numLandmark(0), numStage(0), augnumber(0), augnumShift(0), augnumRotate(0), augnumScale(0), flipflag(0)
		, binaryLength(0), maxNumTrees(0), numTreeDataRatio(0)
	{
		numStage = 4;

		augnumber = (augnumShift + 1) * (augnumRotate + 1) * (augnumScale + 1);
		flipflag = 1;
		for (int i = 0; i < sizeof(validPointsIdx) / sizeof(validPointsIdx[0]); i++)
			validPointsIdx[i] = i;

		binaryLength = 8;
		maxNumTrees = 5;
		numTreeDataRatio = 0.4;

		int initMaxFeatures[] = { 1000, 1000, 1000, 500, 500, 500, 400, 400, 300, 300 };
		maxNumFeatures.resize(sizeof(initMaxFeatures) / sizeof(initMaxFeatures[0]));
		std::copy(initMaxFeatures, initMaxFeatures + sizeof(initMaxFeatures) / sizeof(initMaxFeatures[0]), maxNumFeatures.begin());

		LBF_DATA initMaxRatioRadius[] = { 0.4, 0.3, 0.2, 0.15, 0.12, 0.10, 0.08, 0.06, 0.06, 0.05 };
		maxRatioRadius.resize(sizeof(initMaxRatioRadius) / sizeof(initMaxRatioRadius[0]));
		std::copy(initMaxRatioRadius, initMaxRatioRadius + sizeof(initMaxRatioRadius) / sizeof(initMaxRatioRadius[0]), maxRatioRadius.begin());

		//radius, angles grid initialization
		for (int i = 0; i <= 30; i++)
			radius.push_back((LBF_DATA)i / 30);
		for (int i = 0; i <= 36; i++)
			angles.push_back((LBF_DATA)2 * PI * i / 36);
	}

	~_Params()
	{
	}

} Params;

typedef struct _Data
{
	cv::Mat* srcImgGray;
	Shape gtShape;
	Bbox gtBbox;
	Bbox facedetBbox;

	std::vector<std::vector<Shape> > intermediateShapes;
	std::vector<std::vector<Bbox> > intermediateBboxes;
	std::vector<Shape> shapeResidual;
	std::vector<cv::Mat> tf2meanShape;
	std::vector<cv::Mat> meanShape2tf;

	int originalWidth;
	int originalHeight;
	int curWidth;
	int curHeight;

	_Data()		//default constructor
		: srcImgGray(nullptr)
	{
		srcImgGray = new cv::Mat();
	}
	_Data(const _Data &Other)	//copy constructor
	{
		srcImgGray = new cv::Mat();
		*srcImgGray = Other.srcImgGray->clone();
		gtShape = Other.gtShape;
		gtBbox = Other.gtBbox;
		facedetBbox = Other.facedetBbox;
		intermediateShapes = Other.intermediateShapes;
		intermediateBboxes = Other.intermediateBboxes;
		shapeResidual = Other.shapeResidual;
		for (int i = 0; i < Other.tf2meanShape.size(); i++)	// cv::Mat dosen't provide automatic deep-copy probably...
		{
			tf2meanShape.push_back(Other.tf2meanShape[i].clone());
		}
		for (int i = 0; i < Other.meanShape2tf.size(); i++)
		{
			meanShape2tf.push_back(Other.meanShape2tf[i].clone());
		}
		originalWidth = Other.originalWidth;
		originalHeight = Other.originalHeight;
		curWidth = Other.curWidth;
		curHeight = Other.curHeight;
	}

	~_Data()
	{
		if (srcImgGray)
		{
			delete srcImgGray;
			srcImgGray = nullptr;
		}
	}
} Data;

#endif
