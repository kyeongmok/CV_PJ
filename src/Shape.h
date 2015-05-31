#pragma once

#ifndef __SHAPE_H
#define __SHAPE_H

#ifndef __LBFTYPES_H
#include "LBFTypes.h"
#endif

//openCV library
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

//constants definitions

#define	PI		3.141592653589793238462643383279502884L

//Shape type definitions
typedef cv::Rect Bbox;

class Shape
{
public:
	//Constructors, Destructor
	Shape(void);
	Shape(size_t size);
	~Shape(void);

	//initializing functions
	void LoadShape(std::string shapePath);

	//Member handling functions
	inline size_t size() const				{ return m_Shape.size(); }
	inline void resize(size_t length)		{ m_Shape.resize(length); }
	inline void clear()						{ m_Shape.clear(); }
	inline std::vector<LBF_POINT>& data()	{ return m_Shape; }			//to access data memory directly
	inline LBF_POINT at(size_t i)	const	{ return (i < m_Shape.size()) ? m_Shape[i] : LBF_POINT(-1, -1); }		//to get copy of data value
	Shape clone()	const;

	//static member functions
	Bbox static EnlargingBbox(Bbox srcBbox, LBF_DATA scale, int imgRow, int imgCol);

	//Shape handling functions
	Bbox GetBbox(int imgRow, int imgCol) const;
	void GetMinMaxPoint(LBF_POINT& minPoint, LBF_POINT& maxPoint) const;
	LBF_POINT GetMeanPoint();
	LBF_POINT GetSquareMeanPoint();
	LBF_POINT GetMeanUsedPoint(const std::vector<int>& usedIdxs);
	LBF_POINT GetSquareMeanUsedPoint(const std::vector<int>& usedIdxs);
	LBF_POINT GetVariance();

	void DrawShape(cv::Mat& srcImg, cv::Scalar color);
	Shape RegularizeShape(Bbox refBbox, float scale = 1.0);
	Shape ScaleShape(LBF_DATA scale);
	Shape FlipShape();
	Shape ResetShape(Bbox gtBbox);
	Shape RotateShape(LBF_DATA angleRange = 60);
	Shape TranslateShape(const Shape& meanShape);
	void ToTransMat(cv::Mat& dstTransMat);
	void FromTransMat(const cv::Mat& srcMat);

private:
	std::vector<LBF_POINT> m_Shape;

};
#endif
