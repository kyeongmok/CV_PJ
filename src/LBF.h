#ifndef __LBF_H
#define __LBF_H
#endif

#ifndef __LBFDEF_H
#include "LBFDef.h"
#endif

#ifndef __DECISIONTREE_H
#include "DecisionTree.h"
#endif

#ifndef __SHAPE_H
#include "Shape.h"
#endif

typedef struct _RegressionModel
{
	std::vector<std::vector<std::vector<DecisionTree>>> RandomForests;
	std::vector<cv::Mat> Ws;
} RegressionModel;


class LBF
{
public:
	LBF();
	~LBF();
	
	/* utility functions for other class */
	inline std::vector<Data>& GetData()	{ return m_vData; }
	static void GetGeometricTransform(const cv::Mat& srcMat, const cv::Mat& dstMat, cv::Mat& resultMat);
	static void DoGeometricTransform(const cv::Mat& Transformer, const cv::Mat& srcMat, cv::Mat& dstMat);

	/* API functions */
	/* 1. Common init functions */
	void SetLBFParams(int _binaryLength, int _numStage, int _flipflag, int _maxNumTrees, LBF_DATA _numTreeDataRatio);
	void LoadDatasets(std::string datasetPath, std::vector<std::string> dbnames, std::string initialShapeFileName, std::string pathListFileName);

	/* 2. for train */
	void InitTrainset();
	void Train();
	void SaveModel(std::string rfFilePath, std::string wsFilePath);

	/* 3. for Test */
	void InitTestset();
	void InitFrame(std::string initShapePath);
	void LoadModel(std::string rfFilePath, std::string wsFilePath);
	void Test();	//for test multiple testsets at a time
	void Prediction(const cv::Mat& srcImg, const Bbox srcBbox, Shape& predShape);

private:
	Params m_Params;	//constructor sets default parameters
	std::vector<Data> m_vData;
	RegressionModel m_Model;

	//private member functions
	Shape CalcMeanShape();
	Shape CalcMeanShape(std::string shapePathListFilePath);
	void GetDistance(LBF_POINT point_a, LBF_POINT point_b, LBF_DATA& distance);
	void GetUsedPoints(int init, int fin, int interval, std::vector<int>& used);
	void SampleRandFeat(int numFeatures, std::vector<std::pair<LBF_DATA, LBF_DATA>>& anglePairs);
	void ComputeError(const std::vector< Shape >& ground_truth_all, const std::vector< Shape >& detected_points_all, std::vector<LBF_DATA>& error_per_image);

	void TrainRandomForest(int stage);
	void DeriveBinaryFeature(int stage, cv::Mat& binFeatures);
	void GlobalRegression(const cv::Mat& binaryfeatures, int stage);
	void GlobalPrediction(const cv::Mat& binaryfeatures, int stage);
};