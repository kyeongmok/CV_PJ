#ifndef __SHAPE_H
#include "Shape.h"
#endif

#ifndef __LBFDEF_H
#include "LBFDef.h"
#endif

#include <random>
#include <iostream>
#include <fstream>

//Constructor
Shape::Shape()
{

}

Shape::Shape(size_t size)
{
	if (m_Shape.size() != 0)
	{
		m_Shape.clear();
	}

	m_Shape.resize(size, LBF_POINT(0,0));
}

//Destructor
Shape::~Shape()
{

}

//initializing functions
void Shape::LoadShape(std::string shapePath)
{
	if (m_Shape.size() != 0)
		m_Shape.clear();

	std::ifstream inFile;
	inFile.open(shapePath.c_str());
	if (inFile.is_open())
	{
		std::string tmp;
		LBF_POINT tmpVec;
		char c = 0;
		while (c != '{')
		{
			inFile.get(c);
		}
		inFile.get(c);

		while (true)
		{
			std::getline(inFile, tmp);
			if (tmp[0] == '}')
				break;

			int midIdx;
			for (midIdx = 0; midIdx < tmp.size(); midIdx++)
			{
				if (tmp[midIdx] == ' ' || tmp[midIdx] == '\t')
					break;
			}
			tmpVec.x = std::stod(tmp.substr(0, midIdx));
			tmpVec.y = std::stod(tmp.substr(midIdx));
			m_Shape.push_back(tmpVec);
			tmp.clear();
		}

		inFile.close();
	}
	else
	{
		//std::cout << "fail : load shape file" << std::endl;
	}

	return;
}

//Member handling functions
Shape Shape::clone()	const
{
	Shape dstShape;
	dstShape.resize(m_Shape.size());
	std::copy(m_Shape.begin(), m_Shape.end(), dstShape.m_Shape.begin());

	return dstShape;
}

//static member functions
Bbox Shape::EnlargingBbox(Bbox srcBbox, LBF_DATA scale, int imgRow, int imgCol)
{
	int x = std::floor(srcBbox.x - srcBbox.width * (scale - 1) / 2);
	int y = std::floor(srcBbox.y - srcBbox.height * (scale - 1) / 2);

	Bbox dstBbox;

	dstBbox.x = cv::max(x, 0);
	dstBbox.y = cv::max(y, 0);

	dstBbox.width = cv::min((int)(scale * srcBbox.width - (dstBbox.x - x)), (imgCol - dstBbox.x));
	dstBbox.height = cv::min((int)(scale * srcBbox.height - (dstBbox.y - y)), (imgRow - dstBbox.y));

	return dstBbox;
}

//Shape handling functions
Bbox Shape::GetBbox(int imgRow, int imgCol) const
{
	Bbox dstBbox;

	if (m_Shape.size() < 1)
		return dstBbox;

	LBF_DATA left, top, right, bottom;
	left = right = m_Shape[0].x;
	top = bottom = m_Shape[0].y;

	for (int i = 1; i < m_Shape.size(); i++)
	{
		left = (m_Shape[i].x < left) ? m_Shape[i].x : left;
		right = (m_Shape[i].x > right) ? m_Shape[i].x : right;
		top = (m_Shape[i].y < top) ? m_Shape[i].y : top;
		bottom = (m_Shape[i].y > bottom) ? m_Shape[i].y : bottom;
	}

	dstBbox.x = (int)left;
	dstBbox.y = (int)top;
	dstBbox.width = cv::min((int)right + 1 - (int)left, imgCol - dstBbox.x);
	dstBbox.height = cv::min((int)bottom + 1 - (int)top, imgRow - dstBbox.y);

	return dstBbox;
}

void Shape::GetMinMaxPoint(LBF_POINT& minPoint, LBF_POINT& maxPoint) const
{
	if (m_Shape.size() < 1)
		return;

	minPoint.x = maxPoint.x = m_Shape[0].x;
	minPoint.y = maxPoint.y = m_Shape[0].y;
	for (int i = 0; i < m_Shape.size(); i++)
	{
		minPoint.x = m_Shape[i].x < minPoint.x ? m_Shape[i].x : minPoint.x;
		maxPoint.x = m_Shape[i].x > maxPoint.x ? m_Shape[i].x : maxPoint.x;
		minPoint.y = m_Shape[i].y < minPoint.y ? m_Shape[i].y : minPoint.y;
		maxPoint.y = m_Shape[i].y > maxPoint.y ? m_Shape[i].y : maxPoint.y;
	}

	return;
}

LBF_POINT Shape::GetMeanPoint()
{
	LBF_POINT meanPoint;

	if (m_Shape.size() < 1)
		return meanPoint;

	meanPoint.x = meanPoint.y = 0;
	for (int i = 0; i < m_Shape.size(); i++)
	{
		meanPoint.x += m_Shape[i].x;
		meanPoint.y += m_Shape[i].y;
	}
	meanPoint.x /= m_Shape.size();
	meanPoint.y /= m_Shape.size();

	return meanPoint;
}

LBF_POINT Shape::GetSquareMeanPoint()
{
	LBF_POINT squareMeanPoint;

	if (m_Shape.size() < 1)
		return squareMeanPoint;

	squareMeanPoint.x = squareMeanPoint.y = 0;
	for (int i = 0; i < m_Shape.size(); i++)
	{
		squareMeanPoint.x += m_Shape[i].x * m_Shape[i].x;
		squareMeanPoint.y += m_Shape[i].y * m_Shape[i].y;
	}
	squareMeanPoint.x /= m_Shape.size();
	squareMeanPoint.y /= m_Shape.size();

	return squareMeanPoint;
}

LBF_POINT Shape::GetMeanUsedPoint(const std::vector<int>& usedIdxs)
{
	LBF_POINT meanUsedPoint;

	if (m_Shape.size() < 1)
		return meanUsedPoint;
	
	meanUsedPoint.x = meanUsedPoint.y = 0;
	for (int i = 0; i < usedIdxs.size(); i++)
	{
		meanUsedPoint.x += m_Shape[usedIdxs[i]].x;
		meanUsedPoint.y += m_Shape[usedIdxs[i]].y;
	}
	meanUsedPoint.x /= usedIdxs.size();
	meanUsedPoint.y /= usedIdxs.size();

	return meanUsedPoint;
}

LBF_POINT Shape::GetSquareMeanUsedPoint(const std::vector<int>& usedIdxs)
{
	LBF_POINT squareMeanUsedPoint;

	if (m_Shape.size() < 1)
		return squareMeanUsedPoint;
	
	squareMeanUsedPoint.x = squareMeanUsedPoint.y = 0;

	for (int i = 0; i < usedIdxs.size(); i++)
	{
		squareMeanUsedPoint.x += m_Shape[usedIdxs[i]].x * m_Shape[usedIdxs[i]].x;
		squareMeanUsedPoint.y += m_Shape[usedIdxs[i]].y * m_Shape[usedIdxs[i]].y;
	}
	squareMeanUsedPoint.x /= usedIdxs.size();
	squareMeanUsedPoint.y /= usedIdxs.size();

	return squareMeanUsedPoint;
}

LBF_POINT Shape::GetVariance()
{
	if (m_Shape.size() < 1)
		return LBF_POINT(0, 0);

	LBF_POINT meanPoint = GetMeanPoint();
	LBF_POINT squareMeanPoint = GetSquareMeanPoint();
	return LBF_POINT(squareMeanPoint.x - (meanPoint.x * meanPoint.x), squareMeanPoint.y - (meanPoint.y * meanPoint.y));
}


void Shape::DrawShape(cv::Mat& srcImg, cv::Scalar color)
{
/*	<color parameter>
 *	Usage => cv::Scalar(B, G, R)
 *	(ex)
 *	BLACK = cv::Scalar(0, 0, 0)
 *	WHITE = cv::Scalar(255, 255, 255) ...
 */
	if (m_Shape.size() < 1)
		return;

	for (int i = 0; i < m_Shape.size(); i++)
	{
		cv::circle(srcImg, m_Shape[i], 2, color, -1);
	}
	return;
}

Shape Shape::RegularizeShape(Bbox refBbox, float scale)
{
	Shape dstShape;
	
	if (m_Shape.size() < 1)
		return dstShape;

	dstShape.resize(m_Shape.size());

	LBF_POINT minPoint, maxPoint;
	minPoint = LBF_POINT(refBbox.x, refBbox.y);
	maxPoint = LBF_POINT(refBbox.x + refBbox.width, refBbox.y + refBbox.height);
	for (int i = 0; i < m_Shape.size(); i++)
	{
		LBF_POINT tmpPoint = m_Shape[i];
		tmpPoint -= minPoint;
		tmpPoint.x /= (maxPoint - minPoint).x;
		tmpPoint.y /= (maxPoint - minPoint).y;
		dstShape.data()[i] = tmpPoint * 1000;
	}

	return dstShape;
}

Shape Shape::ScaleShape(LBF_DATA scale)
{
	Shape dstShape;

	if (m_Shape.size() < 1)
		return dstShape;

	dstShape.resize(m_Shape.size());

	LBF_POINT meanPoint = GetMeanPoint();

	for (int i = 0; i < m_Shape.size(); i++)
	{
		dstShape.data()[i].x = (m_Shape[i].x - meanPoint.x) * scale + meanPoint.x;
		dstShape.data()[i].y = (m_Shape[i].y - meanPoint.y) * scale + meanPoint.y;
	}

	return dstShape;
}

Shape Shape::FlipShape()
{
	Shape dstShape;

	if (m_Shape.size() != 68)
	{
		//std::cout << "fail : Invalid arguments to FlipShape" << std::endl;
		return dstShape;
	}

	dstShape.resize(m_Shape.size());

	//flip check
	for (int i = 0; i < 17; i++)
	{
		dstShape.m_Shape[i] = m_Shape[16 - i];
	}

	//flip eyebows
	for (int i = 17; i < 27; i++)
	{
		dstShape.m_Shape[i] = m_Shape[43 - i];
	}

	//flip eyes
	for (int i = 31; i < 36; i++)
	{
		dstShape.m_Shape[i] = m_Shape[66 - i];
	}
	for (int i = 36; i < 46; i++)
	{
		dstShape.m_Shape[i] = m_Shape[81 - i];
	}
	dstShape.m_Shape[40] = m_Shape[47];
	dstShape.m_Shape[41] = m_Shape[46];
	dstShape.m_Shape[46] = m_Shape[41];
	dstShape.m_Shape[47] = m_Shape[40];

	//flip mouth
	for (int i = 48; i < 55; i++)
	{
		dstShape.m_Shape[i] = m_Shape[102 - i];
	}
	for (int i = 55; i < 60; i++)
	{
		dstShape.m_Shape[i] = m_Shape[114 - i];
	}
	for (int i = 60; i < 65; i++)
	{
		dstShape.m_Shape[i] = m_Shape[124 - i];
	}
	for (int i = 65; i < 68; i++)
	{
		dstShape.m_Shape[i] = m_Shape[132 - i];
	}

	return dstShape;
}

Shape Shape::ResetShape(Bbox gtBbox)
{
	Shape dstShape;

	if (m_Shape.size() < 1)
		return dstShape;

	dstShape.resize(m_Shape.size());

	LBF_POINT minPoint, maxPoint;
	GetMinMaxPoint(minPoint, maxPoint);
	LBF_DATA srcWidth = maxPoint.x - minPoint.x;
	LBF_DATA srcHeight = maxPoint.y - minPoint.y;

	for (int i = 0; i < m_Shape.size(); i++)
	{
		dstShape.m_Shape[i].x = (m_Shape[i].x - minPoint.x) * (gtBbox.width / srcWidth) + gtBbox.x;
		dstShape.m_Shape[i].y = (m_Shape[i].y - minPoint.y) * (gtBbox.height / srcHeight) + gtBbox.y;
	}

	return dstShape;
}

Shape Shape::RotateShape(LBF_DATA angleRange)
{
	Shape dstShape;

	if (m_Shape.size() < 1)
		return dstShape;

	dstShape.resize(m_Shape.size());

	std::random_device randDev;
	std::mt19937 generator(randDev());
	std::uniform_real_distribution<LBF_DATA> distr(0, 1);

	LBF_DATA rotAngle = 2 * angleRange * distr(generator) - angleRange;
	LBF_POINT rotCenter = GetMeanPoint();
	LBF_DATA sinT = std::sin(rotAngle * PI / 180);
	LBF_DATA cosT = std::cos(rotAngle * PI / 180);
	for (int i = 0; i < m_Shape.size(); i++)
	{
		dstShape.m_Shape[i].x = rotCenter.x + cosT*(m_Shape[i].x - rotCenter.x) - sinT*(m_Shape[i].y - rotCenter.y);
		dstShape.m_Shape[i].y = rotCenter.y + sinT*(m_Shape[i].x - rotCenter.x) + cosT*(m_Shape[i].y - rotCenter.y);
	}

	return dstShape;
}

Shape Shape::TranslateShape(const Shape& meanShape)
{
	Shape dstShape;

	if (meanShape.m_Shape.size() < 1 || m_Shape.size() < 1)
		return dstShape;

	dstShape.resize(m_Shape.size());

	LBF_POINT srcMinPoint, srcMaxPoint;
	GetMinMaxPoint(srcMinPoint, srcMaxPoint);
	LBF_DATA srcWidth = srcMaxPoint.x - srcMinPoint.x;
	LBF_DATA srcHeight = srcMaxPoint.y - srcMinPoint.y;

	LBF_POINT meanMinPoint, meanMaxPoint;
	meanShape.GetMinMaxPoint(meanMinPoint, meanMaxPoint);
	LBF_DATA meanWidth = meanMaxPoint.x - meanMinPoint.x;
	LBF_DATA meanHeight = meanMaxPoint.y - meanMinPoint.y;

	for (int i = 0; i < m_Shape.size(); i++)
	{
		dstShape.m_Shape[i].x = m_Shape[i].x - srcMinPoint.x;	//srcMeanPoint.x / srcWidth
		dstShape.m_Shape[i].y = m_Shape[i].y - srcMinPoint.y;	//srcMeanPoint.y / srcHeight
	}

	LBF_POINT srcMeanPoint = dstShape.GetMeanPoint();

	for (int i = 0; i < m_Shape.size(); i++)
	{
		dstShape.m_Shape[i].x = meanShape.m_Shape[i].x + meanWidth * (srcMeanPoint.x / srcWidth - 0.5);
		dstShape.m_Shape[i].y = meanShape.m_Shape[i].y + meanHeight * (srcMeanPoint.y / srcHeight - 0.5);
	}

	return dstShape;
}

void Shape::ToTransMat(cv::Mat& dstTransMat)
{
	dstTransMat.release();
	dstTransMat.create(m_Shape.size(), 3, LBF_MAT_TYPE);
	for (int i = 0; i < dstTransMat.rows; i++)
	{
		dstTransMat.at<LBF_DATA>(i, 0) = m_Shape[i].x;
		dstTransMat.at<LBF_DATA>(i, 1) = m_Shape[i].y;
		dstTransMat.at<LBF_DATA>(i, 2) = 1;
	}
	return;
}

void Shape::FromTransMat(const cv::Mat& srcMat)
{
	m_Shape.clear();
	m_Shape.resize(srcMat.rows);

	for (int i = 0; i < srcMat.rows; i++)
	{
		m_Shape[i].x = srcMat.at<LBF_DATA>(i, 0);
		m_Shape[i].y = srcMat.at<LBF_DATA>(i, 1);
	}

	return;
}
