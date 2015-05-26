#include <random>
#include <fstream>
#include <math.h>
#include <iomanip>

#include "LBF.h"
#include "Liblinear.h"
#include "opencv2/objdetect/objdetect.hpp"

//Constructor
LBF::LBF()
{
	m_vData.clear();
}

//Destructor
LBF::~LBF()
{

}


//API functions
void LBF::GetGeometricTransform(const cv::Mat& srcMat, const cv::Mat& dstMat, cv::Mat& resultMat)
{
	/* To get Transform Matrix 'T' from over-determined linear system,
	* We estimate 'T' by Linear Regression method. Such that,
	* V · T = V'
	*	where,
	*		V : n x p source matrix
	*		V': n x q destination matrix
	*		T : p x q transform matrix
	*/

	cv::Mat convSrcMat;
	cv::Mat convDstMat;
	if (srcMat.type() != LBF_MAT_TYPE)
		srcMat.convertTo(convSrcMat, LBF_MAT_TYPE);
	else
		convSrcMat = srcMat;
	if (dstMat.type() != LBF_MAT_TYPE)
		dstMat.convertTo(convDstMat, LBF_MAT_TYPE);
	else
		convDstMat = dstMat;

	cv::Mat tmpRes;

	cv::Mat tmpMat(convSrcMat.cols, convSrcMat.rows, convSrcMat.type());
	cv::transpose(convSrcMat, tmpMat);

	cv::Mat tmptmpMat = (tmpMat * convSrcMat).inv();
	tmpMat = (tmpMat * convSrcMat).inv() * tmpMat;

	int cnt = 0;
	for (int r = 0; r < tmptmpMat.rows; r++)
	{
		for (int c = 0; c < tmptmpMat.cols; c++)
		{
			if (tmptmpMat.at<LBF_DATA>(r, c) != 0)
				cnt++;
		}
	}

	cv::Mat dstTrans(convDstMat.cols, convDstMat.rows, convDstMat.type());
	cv::transpose(convDstMat, dstTrans);
	cv::Mat tmpTrans(tmpMat.cols, tmpMat.rows, tmpMat.type());
	cv::transpose(tmpMat, tmpTrans);

	cv::Mat resultVec;
	for (int i = 0; i < convDstMat.cols; i++)
	{
		resultVec = dstTrans.row(i) * tmpTrans;
		tmpRes.push_back(resultVec);
	}

	cv::transpose(tmpRes, tmpRes);

	resultMat.release();
	resultMat = tmpRes.clone();

	return;
}

void LBF::DoGeometricTransform(const cv::Mat& Transformer, const cv::Mat& srcMat, cv::Mat& dstMat)
{
	cv::Mat tmpRes = srcMat * Transformer;

	dstMat.release();

	dstMat = tmpRes.clone();

	return;
}

void LBF::SetLBFParams(int _binaryLength, int _numStage, int _flipflag, int _maxNumTrees, LBF_DATA _numTreeDataRatio)
{
	int stageLimit = cv::min(m_Params.maxNumFeatures.size(), m_Params.maxRatioRadius.size());
	assert(_numStage <= stageLimit);

	m_Params.binaryLength = _binaryLength;
	m_Params.numStage = _numStage;
	m_Params.flipflag = _flipflag;
	m_Params.maxNumTrees = _maxNumTrees;
	m_Params.numTreeDataRatio = _numTreeDataRatio;
}

void LBF::LoadDatasets(std::string datasetPath, std::vector<std::string> dbnames, std::string initialShapeFileName, std::string pathListFileName)
{
/*	<<<Dataset Directory Structure>>>
 *	datasetPath\\
 *	datasetPath\\initialShapeFileName
 *	datasetPath\\dbname\\
 *	datasetPath\\dbname\\Path_Images.txt
 */

	cv::CascadeClassifier faceDetector;
	faceDetector.load("..\\Data\\haarcascade_frontalface_default.xml");

	//exception : COFW dataset cannot be combined with others
	assert(!(dbnames.size() > 1 && std::find(dbnames.begin(), dbnames.end(), "COFW") != dbnames.end()));

	std::cout << "Load datasets" << std::endl;

	if (m_vData.size() != 0)
		m_vData.clear();

	for (int i = 0; i < dbnames.size(); i++)
	{
		std::string imgPathListFile = datasetPath + dbnames[i] + "\\" + pathListFileName;
		std::vector<std::string> imgPathList;

		std::ifstream inFile;
		inFile.open(imgPathListFile.c_str());
		if (inFile.is_open())
		{
			while (!inFile.eof())
			{
				std::string tmpStr;
				std::getline(inFile, tmpStr);
				if (tmpStr != "")
					imgPathList.push_back(tmpStr);
			}
			inFile.close();
		}
		else
		{
			//std::cout << "fail : load image path list file" << std::endl;
			return;
		}

		for (int j = 0; j < imgPathList.size(); j++)
		{
			Data data;
			cv::Mat curImg = cv::imread(imgPathList[j].c_str(), CV_LOAD_IMAGE_GRAYSCALE);
			std::string shapePath = imgPathList[j].substr(0, imgPathList[j].size() - 3) + "pts";

			data.originalWidth = curImg.cols;
			data.originalHeight = curImg.rows;
			data.gtShape.LoadShape(shapePath);
			data.gtBbox = data.gtShape.GetBbox(data.originalHeight, data.originalWidth);

			Bbox enlargedRegion = data.gtShape.EnlargingBbox(data.gtBbox, 2.0, data.originalHeight, data.originalWidth);

			cv::Mat cropRef(curImg, enlargedRegion);
			*data.srcImgGray = cropRef.clone();

			for (int k = 0; k < data.gtShape.size(); k++)
			{
				(data.gtShape.data())[k].x -= enlargedRegion.x;
				(data.gtShape.data())[k].y -= enlargedRegion.y;
			}
			data.gtBbox.x -= enlargedRegion.x;
			data.gtBbox.y -= enlargedRegion.y;


			//data.facedetBbox = data.gtBbox;
			std::vector<Bbox> vFace;
			faceDetector.detectMultiScale(curImg, vFace, 1.1, 3, CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));

			/* Validation Checks */
			if (vFace.size() < 1)
			{
				std::cout << "Face detection error : numFace is " << vFace.size() << std::endl;
				continue;
			}

			int isValidBbox = false;
			for (int k = 0; k < vFace.size(); k++)
			{
				data.gtBbox.x += enlargedRegion.x;
				data.gtBbox.y += enlargedRegion.y;
				data.facedetBbox = vFace[k];

				int l = cv::max(data.facedetBbox.x, data.gtBbox.x);
				int t = cv::max(data.facedetBbox.y, data.gtBbox.y);
				int r = cv::min(data.facedetBbox.x + data.facedetBbox.width, data.gtBbox.x + data.gtBbox.width);
				int b = cv::min(data.facedetBbox.y + data.facedetBbox.height, data.gtBbox.y + data.gtBbox.height);
				LBF_DATA gtRatio = (LBF_DATA)(r - l) * (b - t) / data.gtBbox.area();
				LBF_DATA fdRatio = (LBF_DATA)(r - 1) * (b - t) / data.facedetBbox.area();
				
				data.gtBbox.x -= enlargedRegion.x;
				data.gtBbox.y -= enlargedRegion.y;
				if (gtRatio > 0.5 && fdRatio > 0.5)
				{
					data.facedetBbox.x -= enlargedRegion.x;
					data.facedetBbox.y -= enlargedRegion.y;
					isValidBbox = true;
					break;
				}
			}

			if (!isValidBbox)
			{
				std::cout << "Face detection error : No valid region is found" << std::endl;
				continue;
			}

			cv::Point lb2(data.facedetBbox.x + data.facedetBbox.width,
				data.facedetBbox.y + data.facedetBbox.height);
			cv::Point tr2(data.facedetBbox.x, data.facedetBbox.y);
			cv::rectangle(*data.srcImgGray, lb2, tr2, cv::Scalar(255, 0, 0), 3, 4, 0);
			cv::Point lb(data.gtBbox.x + data.gtBbox.width,
				data.gtBbox.y + data.gtBbox.height);
			cv::Point tr(data.gtBbox.x, data.gtBbox.y);
			cv::rectangle(*data.srcImgGray, lb, tr, cv::Scalar(0, 255, 0), 3, 4, 0);
		
			cv::imshow("tmp", *data.srcImgGray);
			cv::waitKey();
			
			data.gtBbox = data.facedetBbox;


			data.curWidth = data.srcImgGray->cols;
			data.curHeight = data.srcImgGray->rows;

			m_vData.push_back(data);
		}
	}

	std::string initShapePath = datasetPath + initialShapeFileName;
	m_Params.meanShape.LoadShape(initShapePath);
	//m_Params.meanShape = CalcMeanShape();
	m_Params.numLandmark = m_Params.meanShape.size();
}

void LBF::InitTrainset()
{
	std::cout << "Data Augmentation and Init" << std::endl;
	if (m_Params.flipflag)
	{
		int curSize = m_vData.size();
		for (int i = 0; i < curSize; i++)
		{
			Data data;
			cv::Mat flipImg(m_vData[i].srcImgGray->rows, m_vData[i].srcImgGray->cols, m_vData[i].srcImgGray->type());
			cv::flip(*m_vData[i].srcImgGray, flipImg, 1);
			*data.srcImgGray = flipImg.clone();
			data.originalWidth = m_vData[i].originalWidth;
			data.originalHeight = m_vData[i].originalHeight;
			data.curWidth = m_vData[i].curWidth;
			data.curHeight = m_vData[i].curHeight;

			data.gtShape = m_vData[i].gtShape.FlipShape();
			for (int j = 0; j < data.gtShape.size(); j++)
			{
				data.gtShape.data()[j].x = m_vData[i].curWidth - data.gtShape.data()[j].x;
			}

			data.gtBbox = m_vData[i].gtBbox;
			data.gtBbox.x = data.curWidth - data.gtBbox.x - data.gtBbox.width;

			data.facedetBbox = m_vData[i].facedetBbox;
			data.facedetBbox.x = data.curWidth - data.facedetBbox.x - data.facedetBbox.width;

			m_vData.push_back(data);
		}
 	}

	for (int i = 0; i < m_vData.size(); i++)
	{
		Shape tmpShape;
		for (int j = 0; j < sizeof(m_Params.validPointsIdx) / sizeof(m_Params.validPointsIdx[0]); j++)
		{
			tmpShape.data().push_back(m_vData[i].gtShape.data()[m_Params.validPointsIdx[j]]);
		}
		m_vData[i].gtShape = tmpShape;
		//m_vData[i].gtBbox = m_vData[i].gtShape.GetBbox(m_vData[i].originalHeight, m_vData[i].originalWidth);

		tmpShape.clear();

// 		Shape shapeFacedet = m_Params.meanShape.ResetShape(m_vData[i].facedetBbox);
// 		for (int j = 0; j < sizeof(m_Params.validPointsIdx) / sizeof(m_Params.validPointsIdx[0]); j++)
// 		{
// 			tmpShape.data().push_back(shapeFacedet.data()[m_Params.validPointsIdx[j]]);
// 		}
// 		m_vData[i].facedetBbox = tmpShape.GetBbox(m_vData[i].originalHeight, m_vData[i].originalWidth);
	}
	Shape tmpMeanShape;
	for (int i = 0; i < sizeof(m_Params.validPointsIdx) / sizeof(m_Params.validPointsIdx[0]); i++)
	{
		tmpMeanShape.data().push_back(m_Params.meanShape.data()[m_Params.validPointsIdx[i]]);
	}
	m_Params.meanShape = tmpMeanShape;


	std::random_device randDev;
	std::mt19937 generator(randDev());
	std::uniform_real_distribution<LBF_DATA> distr(0, 1);

	for (int i = 0; i < m_vData.size(); i++)
	{
		std::vector<int> indice_rotate(m_Params.augnumber);
		std::vector<int> indice_shift(m_Params.augnumber);
		std::vector<LBF_DATA> scales(m_Params.augnumber);
		for (int j = 0; j < m_Params.augnumber; j++)
		{
			indice_rotate[j] = ceil(distr(generator) * m_vData.size());
			indice_shift[j] = ceil(distr(generator) * m_vData.size());
			scales[j] = 1 + 0.2 * (distr(generator) - 0.5);
		}

		m_vData[i].intermediateShapes.resize(m_Params.numStage + 1, std::vector<Shape>(m_Params.augnumber));
		m_vData[i].intermediateBboxes.resize(m_Params.numStage + 1, std::vector<Bbox>(m_Params.augnumber));
		m_vData[i].shapeResidual.resize(m_Params.augnumber);
		m_vData[i].tf2meanShape.resize(m_Params.augnumber);
		m_vData[i].meanShape2tf.resize(m_Params.augnumber);

		for (int sr = 0; sr < m_Params.augnumber; sr++)
		{
			Shape meanShapeResized = m_Params.meanShape.ResetShape(m_vData[i].facedetBbox);
			if (sr != 0)
			{
				if (m_Params.augnumScale)
				{
					meanShapeResized = meanShapeResized.ScaleShape(scales[sr]);
				}

				if (m_Params.augnumRotate)
				{
					meanShapeResized = meanShapeResized.RotateShape();
				}

				if (m_Params.augnumShift)
				{
					meanShapeResized = m_vData[indice_shift[sr]].gtShape.TranslateShape(meanShapeResized);
				}
			}

			m_vData[i].intermediateShapes[0][sr] = meanShapeResized;
			m_vData[i].intermediateBboxes[0][sr] = (sr == 0) ? m_vData[i].facedetBbox : meanShapeResized.GetBbox(m_vData[i].originalHeight, m_vData[i].originalWidth);

			meanShapeResized = m_Params.meanShape.ResetShape(m_vData[i].intermediateBboxes[0][sr]);

			Shape tfLocalShape(m_vData[i].intermediateShapes[0][sr].size());
			Shape meanLocalShape(meanShapeResized.size());
			LBF_POINT tfPoint = m_vData[i].intermediateShapes[0][sr].GetMeanPoint();
			LBF_POINT meanPoint = meanShapeResized.GetMeanPoint();

			for (int j = 0; j < tfLocalShape.size(); j++)
			{
				tfLocalShape.data()[j] = m_vData[i].intermediateShapes[0][sr].data()[j] - tfPoint;
				meanLocalShape.data()[j] = meanShapeResized.data()[j] - meanPoint;
			}

			cv::Mat tfLocalMat, meanLocalMat;
			tfLocalShape.ToTransMat(tfLocalMat);
			meanLocalShape.ToTransMat(meanLocalMat);
			GetGeometricTransform(tfLocalMat, meanLocalMat, m_vData[i].tf2meanShape[sr]);
			GetGeometricTransform(meanLocalMat, tfLocalMat, m_vData[i].meanShape2tf[sr]);

			Shape tmpShapeResidual(m_vData[i].gtShape.size());
			for (int j = 0; j < tmpShapeResidual.size(); j++)
			{
				tmpShapeResidual.data()[j].x = (m_vData[i].gtShape.data()[j].x - m_vData[i].intermediateShapes[0][sr].data()[j].x) / m_vData[i].intermediateBboxes[0][sr].width;
				tmpShapeResidual.data()[j].y = (m_vData[i].gtShape.data()[j].y - m_vData[i].intermediateShapes[0][sr].data()[j].y) / m_vData[i].intermediateBboxes[0][sr].height;
			}
			cv::Mat tmpShapeResidualMat, shapeResidualMat;
			tmpShapeResidual.ToTransMat(tmpShapeResidualMat);
			DoGeometricTransform(m_vData[i].tf2meanShape[0], tmpShapeResidualMat, shapeResidualMat);
			m_vData[i].shapeResidual[sr].FromTransMat(shapeResidualMat);
		}
	}
}

void LBF::Train()
{
	m_Model.RandomForests.resize(m_Params.numStage);
	m_Model.Ws.resize(m_Params.numStage);

	for (int i = 0; i < m_Params.numStage; i++)
	{
		std::cout << "<< Train " << i + 1 << " th of " << m_Params.numStage << " stages >>" << std::endl;

		if (m_Model.RandomForests[i].empty())
		{
			std::cout << "train random forests for landmarks..." << std::endl;
			TrainRandomForest(i);
		}

		std::cout << "extract local binary features..." << std::endl;
		cv::Mat binFeatures;
		DeriveBinaryFeature(i, binFeatures);

		std::cout << "learn global regressors..." << std::endl;
		GlobalRegression(binFeatures, i);
	}
}

void LBF::SaveModel(std::string rfFilePath, std::string wsFilePath)
{
	assert(m_Model.RandomForests.size() != 0 && m_Model.Ws.size() != 0);

	////////* Save RandomForest *////////
	std::cout << "saving Random Forest train result" << std::endl;
	std::ofstream outStream;
	outStream.open(rfFilePath);
	outStream << std::setprecision(std::numeric_limits<LBF_DATA>::max_digits10 - 1);

	/* common parameters */
	outStream << "param " << m_Params.binaryLength << std::endl;
	for (int stage = 0; stage < m_Model.RandomForests.size(); stage++)
	{
		/* for each stage */
		outStream << "stage " << stage << std::endl;
		for (int lmark = 0; lmark < m_Model.RandomForests[0].size(); lmark++)
		{
			/* for each landmark */
			outStream << "lmark " << lmark << std::endl;
			for (int curTree = 0; curTree < m_Model.RandomForests[0][0].size(); curTree++)
			{
				/* for each decision tree */
				outStream << "tree " << curTree << std::endl;
				m_Model.RandomForests[stage][lmark][curTree].SaveTree(outStream);
			}
		}
	}
	outStream.close();

	////////* Save Ws *////////
	std::cout << "saving Ws train result" << std::endl;
	cv::FileStorage fs(wsFilePath, cv::FileStorage::WRITE);
	for (int stage = 0; stage < m_Model.Ws.size(); stage++)
	{
		// naming for each Ws
		std::stringstream strStream;
		std::string curWs = "Ws";
		strStream << curWs << stage;
		curWs = strStream.str();

		fs << curWs << m_Model.Ws[stage];
	}
	fs.release();

	return;
}

void LBF::InitTestset()
{
	std::cout << "Data Init" << std::endl;
	m_Params.augnumber = 1;		//testsets do not augmentation
	for (int i = 0; i < m_vData.size(); i++)
	{
		m_vData[i].intermediateShapes.resize(m_Params.numStage + 1, std::vector<Shape>(1));
		m_vData[i].intermediateBboxes.resize(m_Params.numStage + 1, std::vector<Bbox>(1));
		m_vData[i].shapeResidual.resize(1);
		m_vData[i].tf2meanShape.resize(1);
		m_vData[i].meanShape2tf.resize(1);

		Shape meanShapeResized = m_Params.meanShape.ResetShape(m_vData[i].facedetBbox);

		m_vData[i].intermediateShapes[0][0] = meanShapeResized;
		m_vData[i].intermediateBboxes[0][0] = m_vData[i].facedetBbox;

		meanShapeResized = m_Params.meanShape.ResetShape(m_vData[i].intermediateBboxes[0][0]);

		Shape tfLocalShape(m_vData[i].intermediateShapes[0][0].size());
		Shape meanLocalShape(meanShapeResized.size());
		LBF_POINT tfPoint = m_vData[i].intermediateShapes[0][0].GetMeanPoint();
		LBF_POINT meanPoint = meanShapeResized.GetMeanPoint();

		for (int j = 0; j < tfLocalShape.size(); j++)
		{
			tfLocalShape.data()[j] = m_vData[i].intermediateShapes[0][0].data()[j] - tfPoint;
			meanLocalShape.data()[j] = meanShapeResized.data()[j] - meanPoint;
		}

		cv::Mat tfLocalMat, meanLocalMat;
		tfLocalShape.ToTransMat(tfLocalMat);
		meanLocalShape.ToTransMat(meanLocalMat);
		GetGeometricTransform(tfLocalMat, meanLocalMat, m_vData[i].tf2meanShape[0]);
		GetGeometricTransform(meanLocalMat, tfLocalMat, m_vData[i].meanShape2tf[0]);

// 		Shape tmpShapeResidual(m_vData[i].gtShape.size());
// 		for (int j = 0; j < tmpShapeResidual.size(); j++)
// 		{
// 			tmpShapeResidual.data()[j].x = (m_vData[i].gtShape.data()[j].x - m_vData[i].intermediateShapes[0][0].data()[j].x) / m_vData[i].intermediateBboxes[0][0].width;
// 			tmpShapeResidual.data()[j].y = (m_vData[i].gtShape.data()[j].y - m_vData[i].intermediateShapes[0][0].data()[j].y) / m_vData[i].intermediateBboxes[0][0].height;
// 		}
// 		cv::Mat tmpShapeResidualMat, shapeResidualMat;
// 		tmpShapeResidual.ToTransMat(tmpShapeResidualMat);
// 		DoGeometricTransform(m_vData[i].tf2meanShape[0], tmpShapeResidualMat, shapeResidualMat);
// 		m_vData[i].shapeResidual[0].FromTransMat(shapeResidualMat);
	}
}

void LBF::InitFrame(std::string initShapePath)
{
	m_Params.augnumber = 1;
	m_Params.meanShape.LoadShape(initShapePath);
	m_Params.numLandmark = m_Params.meanShape.size();

	return;
}

void LBF::LoadModel(std::string rfFilePath, std::string wsFilePath)
{
	m_Model.RandomForests.resize(m_Params.numStage, std::vector<std::vector<DecisionTree>>(m_Params.numLandmark, std::vector<DecisionTree>(m_Params.maxNumTrees)));
	m_Model.Ws.resize(m_Params.numStage);

	////////* Load RandomForest *////////
	std::cout << "loading Random Forest train model" << std::endl;
	std::ifstream inStream;
	inStream.open(rfFilePath);
	assert(inStream.is_open());

	std::string labelStr;

	/* common parameters */
	inStream >> labelStr;
	assert(labelStr == "param");
	inStream >> m_Params.binaryLength;
	for (int stage = 0; stage < m_Model.RandomForests.size(); stage++)
	{
		/* for each stage */
		int readStage;
		inStream >> labelStr >> readStage;
		assert(labelStr == "stage" && readStage == stage);
		for (int lmark = 0; lmark < m_Model.RandomForests[0].size(); lmark++)
		{
			/* for each landmark */
			int readLmark;
			inStream >> labelStr >> readLmark;
			assert(labelStr == "lmark" && readLmark == lmark);
			for (int curTree = 0; curTree < m_Model.RandomForests[0][0].size(); curTree++)
			{
				/* for each decision tree */
				int readTree;
				inStream >> labelStr >> readTree;
				assert(labelStr == "tree" && readTree == curTree);
				m_Model.RandomForests[stage][lmark][curTree].LoadTree(inStream);
				m_Model.RandomForests[stage][lmark][curTree].SetDTParams(m_Params.binaryLength);
			}
		}
	}
	inStream.close();

	////////* Load Ws *////////
	std::cout << "loading Ws train model" << std::endl;
	cv::FileStorage fs(wsFilePath, cv::FileStorage::READ);
	for (int stage = 0; stage < m_Model.Ws.size(); stage++)
	{
		// naming for each Ws
		std::stringstream strStream;
		std::string curWs = "Ws";
		strStream << curWs << stage;
		curWs = strStream.str();

		fs[curWs] >> m_Model.Ws[stage];
	}
	fs.release();

	return;
}

void LBF::Test()
{
	for (int i = 0; i < m_Params.numStage; i++)
	{
		std::cout << "extract local binary features..." << std::endl;
		cv::Mat binFeatures;
		DeriveBinaryFeature(i, binFeatures);

		std::cout << "global prediction..." << std::endl;
		GlobalPrediction(binFeatures, i);
	}
}

void LBF::Prediction(const cv::Mat& srcImg, const Bbox srcBbox, Shape& predShape)
{
	predShape.clear();
	cv::Mat srcGrayImg = srcImg.clone();
	if (srcGrayImg.channels() != 1)
		cv::cvtColor(srcGrayImg, srcGrayImg, CV_BGR2GRAY);

	/* Data init */
	std::vector<Shape> intermediateShapes(m_Params.numStage + 1);
	std::vector<Bbox> intermediateBboxes(m_Params.numStage + 1);
	cv::Mat tf2meanShape = (cv::Mat_<LBF_DATA>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);		//Identity Matrix
	cv::Mat meanShape2tf = tf2meanShape.clone();

	Shape meanShapeResized = m_Params.meanShape.ResetShape(srcBbox);
	intermediateShapes[0] = meanShapeResized;
	intermediateBboxes[0] = srcBbox;

	for (int i = 0; i < m_Params.numStage; i++)
	{
		/* Derive Binary Feature */
// 		std::cout << "extract local binary features..." << std::endl;
// 		myClock.MeasureBegin();

		cv::Mat binFeature;
		int totalBinLength = m_Params.binaryLength * m_Params.maxNumTrees * m_Params.numLandmark;
		binFeature.release();
		binFeature.create(1, totalBinLength, LBF_MAT_TYPE);

		std::vector<char> vBinFeature;
		for (int lmark = 0; lmark < m_Params.numLandmark; lmark++)
		{
			for (int tnum = 0; tnum < m_Params.maxNumTrees; tnum++)
			{
				std::vector<char> vTmpBinary;
				m_Model.RandomForests[i][lmark][tnum].DeriveBinFeature(srcGrayImg, m_Params, intermediateBboxes[i], intermediateShapes[i].data()[lmark], meanShape2tf, i,
					vTmpBinary);
				vBinFeature.insert(vBinFeature.end(), vTmpBinary.begin(), vTmpBinary.end());
			}
		}

		assert(vBinFeature.size() == binFeature.cols);
		for (int c = 0; c < vBinFeature.size(); c++)
		{
			binFeature.at<LBF_DATA>(0, c) = (LBF_DATA)vBinFeature[c];
		}

// 		myClock.MeasureEnd();
// 		ntime = myClock.GetSectionTime();
// 		std::cout << "extract binary features time : " << ntime << "s" << std::endl;


		/* Global Prediction */
// 		std::cout << "global prediction..." << std::endl;
// 		myClock.MeasureBegin();

		cv::Mat deltaShapeMat = binFeature * m_Model.Ws[i];
		Shape& curPredShape = intermediateShapes[i];
		Shape& nextPredShape = intermediateShapes[i + 1];

		Shape deltaShapePred(deltaShapeMat.cols / 2);
		for (int j = 0; j < deltaShapePred.size(); j++)
		{
			deltaShapePred.data()[j].x = deltaShapeMat.at<LBF_DATA>(0, j);
			deltaShapePred.data()[j].y = deltaShapeMat.at<LBF_DATA>(0, j + deltaShapePred.size());
		}

		cv::Mat deltaShapePredMat;
		deltaShapePred.ToTransMat(deltaShapePredMat);
		DoGeometricTransform(meanShape2tf, deltaShapePredMat, deltaShapePredMat);
		deltaShapePred.FromTransMat(deltaShapePredMat);

		for (int j = 0; j < deltaShapePred.size(); j++)
		{
			deltaShapePred.data()[j].x = deltaShapePred.data()[j].x * intermediateBboxes[i].width;
			deltaShapePred.data()[j].y = deltaShapePred.data()[j].y * intermediateBboxes[i].height;
		}

		nextPredShape.resize(curPredShape.size());
		for (int j = 0; j < nextPredShape.size(); j++)
		{
			nextPredShape.data()[j] = curPredShape.data()[j] + deltaShapePred.data()[j];
		}
		intermediateBboxes[i + 1] = intermediateBboxes[i];

		Shape meanShapeResized = m_Params.meanShape.ResetShape(intermediateBboxes[i + 1]);
		Shape intermLocalShape(intermediateShapes[i + 1].size());
		Shape meanLocalShape(meanShapeResized.size());
		LBF_POINT intermPoint = intermediateShapes[i + 1].GetMeanPoint();
		LBF_POINT meanPoint = meanShapeResized.GetMeanPoint();
		for (int j = 0; j < intermLocalShape.size(); j++)
		{
			intermLocalShape.data()[j] = intermediateShapes[i + 1].data()[j] - intermPoint;
			meanLocalShape.data()[j] = meanShapeResized.data()[j] - meanPoint;
		}

		cv::Mat intermMat, meanResizedMat;
		intermLocalShape.ToTransMat(intermMat);
		meanLocalShape.ToTransMat(meanResizedMat);
		GetGeometricTransform(intermMat, meanResizedMat, tf2meanShape);
		GetGeometricTransform(meanResizedMat, intermMat, meanShape2tf);

// 		myClock.MeasureEnd();
// 		ntime = myClock.GetSectionTime();
// 		std::cout << "global prediction time : " << ntime << "s" << std::endl << std::endl;
	}

	predShape = intermediateShapes.back();
}


//private member functions
Shape LBF::CalcMeanShape()
{
	Shape meanShape(m_vData[0].gtShape.size());
	for (int i = 0; i < m_vData.size(); i++)
	{
		LBF_POINT minPoint, maxPoint;
		minPoint = LBF_POINT(m_vData[i].gtBbox.x, m_vData[i].gtBbox.y);
		maxPoint = LBF_POINT(m_vData[i].gtBbox.x + m_vData[i].gtBbox.width, m_vData[i].gtBbox.y + m_vData[i].gtBbox.height);
		for (int j = 0; j < meanShape.size(); j++)
		{
			LBF_POINT tmpPoint = m_vData[i].gtShape.data()[j];
			tmpPoint -= minPoint;
			tmpPoint.x /= (maxPoint - minPoint).x;
			tmpPoint.y /= (maxPoint - minPoint).y;
			meanShape.data()[j] += tmpPoint;
		}
	}

	for (int i = 0; i < meanShape.size(); i++)
	{
		meanShape.data()[i].x /= m_vData.size();
		meanShape.data()[i].y /= m_vData.size();
	}

	return meanShape;
}

Shape LBF::CalcMeanShape(std::string shapePathListFilePath)
{
	//assume that shapePathListFilePath has path of each shape line by line
	std::ifstream inFile(shapePathListFilePath.c_str());
	std::string shapePath;
	std::vector<Shape> shapes;
	while (std::getline(inFile, shapePath))
	{
		Shape curr;
		curr.LoadShape(shapePath);
		shapes.push_back(curr);
	}
	inFile.close();

	Shape meanShape(m_vData[0].gtShape.size());
	// assume that each shape contains same number of points
	for (int i = 0; i < shapes.size(); i++)
	{
		LBF_POINT minPoint, maxPoint;
		shapes[i].GetMinMaxPoint(minPoint, maxPoint);

		// move to origin point and resize (seems to be a kind of normalization)
		for (int j = 0; j < shapes[i].size(); j++)
		{
			shapes[i].data()[j] -= minPoint;
			shapes[i].data()[j].x /= (maxPoint - minPoint).x;
			shapes[i].data()[j].y /= (maxPoint - minPoint).y;

			meanShape.data()[j] += shapes[i].data()[j];
		}
	}

	//divide by N to get mean
	int N = shapes.size();
	for (int i = 0; i < meanShape.size(); i++)
	{
		meanShape.data()[i].x /= LBF_POINT(N, N).x;
		meanShape.data()[i].y /= LBF_POINT(N, N).y;
	}

	return meanShape;
}

void LBF::GetDistance(LBF_POINT point_a, LBF_POINT point_b, LBF_DATA& distance)
{
	distance = sqrt((point_a.x - point_b.x) * (point_a.x - point_b.x) + (point_a.y - point_b.y) * (point_a.y - point_b.y));
	return;
}

void LBF::GetUsedPoints(int init, int fin, int interval, std::vector<int>& used)
{
	used.clear();

	for (int i = init; i <= fin; i = i + interval)
		used.push_back(i);
	return;
}

void LBF::SampleRandFeat(int numFeatures, std::vector<std::pair<LBF_DATA, LBF_DATA>>& anglePairs)
{
	if (anglePairs.size() != 0)
	{
		anglePairs.clear();
	}

	anglePairs.resize(numFeatures);

	std::vector<LBF_DATA> thethasA, thethasB;
	thethasA.resize(numFeatures);
	thethasB.resize(numFeatures);
	for (int i = 0; i < numFeatures; i++)
	{
		thethasA[i] = thethasB[i] = 2 * PI * (LBF_DATA)i / (numFeatures - 1);
	}

	std::random_device randDev;
	std::mt19937 generator(randDev());

	std::shuffle(thethasA.begin(), thethasA.end(), generator);
	std::shuffle(thethasB.begin(), thethasB.end(), generator);

	for (int i = 0; i < numFeatures; i++)
	{
		anglePairs[i] = std::make_pair(thethasA[i], thethasB[i]);
	}

	return;
}

void LBF::ComputeError(const std::vector< Shape >& ground_truth_all, const std::vector< Shape >& detected_points_all, std::vector<LBF_DATA>& error_per_image)
{
	//regression할 때 error를 계산해주는 function 인데 이것은 나중에 계산
	/*function[error_per_image] = ComputeError(ground_truth_all, detected_points_all)
	% ComputeError
	%   compute the average point - to - point Euclidean error normalized by the
	%   inter - ocular distance(measured as the Euclidean distance between the
	%   outer corners of the eyes)
	%
	%   Inputs:
	%          grounth_truth_all, size : num_of_points x 2 x num_of_images
	%          detected_points_all, size : num_of_points x 2 x num_of_images
	%   Output :
	%          error_per_image, size : num_of_images x 1*/

	int num_of_images = ground_truth_all.size();
	int num_of_points = ground_truth_all[1].size();

	for (int i = 0; i < num_of_images; i++)
	{
		Shape detected_points = detected_points_all[i];
		Shape ground_truth_points = ground_truth_all[i];
		LBF_DATA interocular_distance = 0.0f;
		if (num_of_points == 68) //  norm((mean(shape_gt(37:42, : )) - mean(shape_gt(43:48, : ))));
		{
			std::vector<int> used1(6), used2(6);
			for (int i = 0; i < used1.size(); i++)
			{
				used1[i] = 36 + i;
				used2[i] = 42 + i;
			}

			LBF_POINT interocular_distance1, interocular_distance2;
			interocular_distance1 = ground_truth_points.GetMeanUsedPoint(used1);
			interocular_distance2 = ground_truth_points.GetMeanUsedPoint(used2);

			interocular_distance = sqrt((interocular_distance1.x - interocular_distance2.x)*(interocular_distance1.x - interocular_distance2.x) + (interocular_distance1.y - interocular_distance2.y)*(interocular_distance1.y - interocular_distance2.y));
		}
		else if (num_of_points == 51)
		{
			interocular_distance = sqrt((ground_truth_points.data()[19].x - ground_truth_points.data()[28].x) * (ground_truth_points.data()[19].x - ground_truth_points.data()[28].x)
				+ (ground_truth_points.data()[19].y - ground_truth_points.data()[19].y) * (ground_truth_points.data()[19].y - ground_truth_points.data()[19].y));
		}
		else if (num_of_points == 29)
		{
			std::vector<int> used1(5), used2(5);
			for (int i = 0; i < used1.size(); i++)
			{
				used1[i] = 8 + 2 * i;
				used2[i] = 9 + 2 * i;
			}

			LBF_POINT interocular_distance1, interocular_distance2;
			interocular_distance1 = ground_truth_points.GetMeanUsedPoint(used1);
			interocular_distance2 = ground_truth_points.GetMeanUsedPoint(used2);

			interocular_distance = sqrt((interocular_distance1.x - interocular_distance2.x)*(interocular_distance1.x - interocular_distance2.x) + (interocular_distance1.y - interocular_distance2.y)*(interocular_distance1.y - interocular_distance2.y));
		}


		else
		{
			std::vector<int> used1(2), used2(2);
			used1[0] = 0; used1[1] = 1;
			used2[0] = ground_truth_points.size() - 2; used2[1] = ground_truth_points.size() - 1;
			LBF_POINT interocular_distance1, interocular_distance2;
			interocular_distance1 = ground_truth_points.GetMeanUsedPoint(used1);
			interocular_distance2 = ground_truth_points.GetMeanUsedPoint(used2);

			interocular_distance = sqrt((interocular_distance1.x - interocular_distance2.x)*(interocular_distance1.x - interocular_distance2.x) + (interocular_distance1.y - interocular_distance2.y)*(interocular_distance1.y - interocular_distance2.y));
		}


		LBF_DATA sum = 0;

		for (int j = 0; j < num_of_points; j++)
		{
			sum += sqrt((detected_points.data()[j].x - ground_truth_points.data()[j].x) * (detected_points.data()[j].x - ground_truth_points.data()[j].x)
				+ (detected_points.data()[j].y - ground_truth_points.data()[j].y) * (detected_points.data()[j].y - ground_truth_points.data()[j].y));
		}
		error_per_image.push_back(sum / (num_of_points * interocular_distance));
	}
	return;
}

void LBF::TrainRandomForest(int stage)
{
	std::vector<std::vector<DecisionTree>>& curRfs = m_Model.RandomForests[stage];
	/* random sampling parameter setting (random forest) */
	int numTreeData = (int)(m_vData.size() * m_Params.numTreeDataRatio);
	std::vector<Data> vTreeData;

	std::random_device randDev;
	std::mt19937 generator(randDev());
	std::uniform_int_distribution<int> distr(0, m_vData.size() - 1);

	/* Start learning */
	curRfs.resize(m_Params.numLandmark, std::vector<DecisionTree>(m_Params.maxNumTrees));
	for (int i = 0; i < m_Params.numLandmark; i++)
	{
		for (int j = 0; j < m_Params.maxNumTrees; j++)
		{
			/* random sampling */
			vTreeData.clear();
			for (int k = 0; k < numTreeData; k++)
			{
				vTreeData.push_back(m_vData[distr(generator)]);
			}

			/* train */
			curRfs[i][j].SetDTParams(m_Params.binaryLength);
			curRfs[i][j].Train(vTreeData, m_Params, stage, i);
		}
	}

	return;
}

void LBF::DeriveBinaryFeature(int stage, cv::Mat& binFeatures)
{
	/* for multiple Images */
	std::vector<std::vector<DecisionTree>>& curRfs = m_Model.RandomForests[stage];
	int totalBinLength = m_Params.binaryLength * m_Params.maxNumTrees * m_Params.numLandmark;
	int numData = m_vData.size() * m_Params.augnumber;
	binFeatures.release();
	binFeatures.create(numData, totalBinLength, LBF_MAT_TYPE);

	for (int i = 0; i < numData; i++)
	{
		int dataIdx = std::floor(i / m_Params.augnumber);
		int augIdx = i % m_Params.augnumber;

		const cv::Mat* curImg = m_vData[dataIdx].srcImgGray;
		const cv::Mat& curMeanshape2tf = m_vData[dataIdx].meanShape2tf[augIdx];
		Bbox curBbox = m_vData[dataIdx].intermediateBboxes[stage][augIdx];
		Shape curShape = m_vData[dataIdx].intermediateShapes[stage][augIdx];

		std::vector<char> vBinFeature;
		for (int lmark = 0; lmark < m_Params.numLandmark; lmark++)
		{
			for (int tnum = 0; tnum < m_Params.maxNumTrees; tnum++)
			{
				std::vector<char> vTmpBinary;
				curRfs[lmark][tnum].DeriveBinFeature(*curImg, m_Params, curBbox, curShape.data()[lmark], curMeanshape2tf, stage,
					vTmpBinary);
				vBinFeature.insert(vBinFeature.end(), vTmpBinary.begin(), vTmpBinary.end());
			}
		}

		assert(vBinFeature.size() == binFeatures.cols);
		for (int c = 0; c < vBinFeature.size(); c++)
		{
			binFeatures.at<LBF_DATA>(i, c) = (LBF_DATA)vBinFeature[c];
		}
	}

	return;
}

void LBF::GlobalRegression(const cv::Mat& binaryfeatures, int stage)
{
	cv::Mat& curW = m_Model.Ws[stage];

	// organize the groundtruth shape
	int dbsize = m_vData.size();
	// concatenate 2 - D coordinates into a vector(N X(2 * L))
	std::vector< std::vector< LBF_DATA > > deltashapes;
	deltashapes.resize(dbsize*m_Params.augnumber);
	for (int i = 0; i < dbsize*m_Params.augnumber; i++)
		deltashapes[i].resize(2 * m_Params.meanShape.size());

	std::vector<LBF_DATA> dist_pupils;
	dist_pupils.resize(dbsize*(m_Params.augnumber));
	std::vector< Shape >gtshapes;
	gtshapes.resize(dbsize*m_Params.augnumber);

	//dist_pupils = zeros(dbsize*(m_Params.augnumber), 1);
	//gtshapes = zeros([size(m_Params.meanshape) dbsize*(m_Params.augnumber)]);  // concatenate 2 - D coordinates into a vector(N X(2 * L))

	for (int i = 0; i < dbsize*(m_Params.augnumber); i++)
	{
		int k = std::floor(i / (m_Params.augnumber));
		int s = i % (m_Params.augnumber);

		Shape shape_gt = m_vData[k].gtShape.clone();

		if (shape_gt.size() == 68)
		{
			LBF_POINT shape_gt_mean1, shape_gt_mean2;
			std::vector<int> used1, used2;
			GetUsedPoints(36, 41, 1, used1);
			shape_gt_mean1 = shape_gt.GetMeanUsedPoint(used1);

			GetUsedPoints(42, 47, 1, used2);
			shape_gt_mean2 = shape_gt.GetMeanUsedPoint(used2);

			GetDistance(shape_gt_mean1, shape_gt_mean2, dist_pupils[i]);
		}

		else if (shape_gt.size() == 51)
			GetDistance(shape_gt.data()[19], shape_gt.data()[28], dist_pupils[i]);

		else if (shape_gt.size() == 29)
		{
			LBF_POINT shape_gt_mean1, shape_gt_mean2;
			std::vector<int> used1, used2;
			GetUsedPoints(8, 16, 2, used1);
			shape_gt_mean1 = shape_gt.GetMeanUsedPoint(used1);

			GetUsedPoints(9, 17, 2, used2);
			shape_gt_mean2 = shape_gt.GetMeanUsedPoint(used2);

			GetDistance(shape_gt_mean1, shape_gt_mean2, dist_pupils[i]);
		}

		else
		{
			LBF_POINT shape_gt_mean1, shape_gt_mean2;
			std::vector<int> used1, used2;
			GetUsedPoints(0, 1, 1, used1);
			shape_gt_mean1 = shape_gt.GetMeanUsedPoint(used1);

			GetUsedPoints(shape_gt.size() - 2, shape_gt.size() - 1, 1, used2);
			shape_gt_mean2 = shape_gt.GetMeanUsedPoint(used2);

			GetDistance(shape_gt_mean1, shape_gt_mean2, dist_pupils[i]);
		}

		gtshapes[i] = shape_gt.clone();

		Shape delta_shape = m_vData[k].shapeResidual[s].clone();

		for (int j = 0; j < delta_shape.size(); j++)
		{
			deltashapes[i][j] = delta_shape.data()[j].x;
			deltashapes[i][j + delta_shape.size()] = delta_shape.data()[j].y;
		}

		/*for (int j = 0; j < delta_shape.size(); j++)
		deltashapes[i].push_back(delta_shape[j].x);
		for (int j = 0; j < delta_shape.size(); j++)
		deltashapes[i].push_back(delta_shape[j].y);*/
	}
	// conduct regression using libliear
	// X : binaryfeatures
	// Y : gtshapes

	//param = sprintf('-s 12 -p 0 -c %f -q heart_scale', 1 / (size(binaryfeatures, 1)));
	curW.release();
	curW.create(binaryfeatures.cols, deltashapes[0].size(), LBF_MAT_TYPE);
	//cv::transpose(W_liblinear, W_liblinear);

	//tic;
	//parellel
	//for (int o = 0; o < deltashapes[0].size(); o++)
	//{
	//	cv::Mat deltaShapesCol(deltashapes.size(), 1, LBF_MAT_TYPE);
	//	for (int j = 0; j < deltashapes.size(); j++)
	//		deltaShapesCol.at<LBF_DATA>(j, 0) = deltashapes[j][o];

	//	//sparse 처리 추가(행렬 계산 빨리 하기 위해)-행렬에서 0인것들은 무시하고 계산(matlab에서 sparse참조)
	//	cv::Mat W_liblinear_col;
	//	GetGeometricTransform(binaryfeatures, deltaShapesCol, W_liblinear_col);

	//	cv::transpose(W_liblinear_col, W_liblinear_col);

	//	W_liblinear.push_back(W_liblinear_col);
	//}

	/* tmp code for binFeatures : vector -> cv::Mat */
	// 	cv::Mat deltashapesMat(deltashapes.size(), deltashapes[0].size(), LBF_MAT_TYPE);
	// 	for (int row = 0; row < deltashapesMat.rows; row++)
	// 	{
	// 		for (int col = 0; col < deltashapesMat.cols; col++)
	// 		{
	// 			deltashapesMat.at<LBF_DATA>(row, col) = deltashapes[row][col];
	// 		}
	// 	}
	//cv::Mat deltashapesMat(deltashapes.size(), deltashapes[0].size(), LBF_MAT_TYPE, deltashapes.data());


	LBF_DATA* deltashapesArray = new LBF_DATA[deltashapes.size() * deltashapes[0].size()];
	for (int r = 0; r < deltashapes.size(); r++)
	{
		for (int c = 0; c < deltashapes[0].size(); c++)
		{
			deltashapesArray[r * deltashapes[0].size() + c] = deltashapes[r][c];
		}
	}

	L2RegularL2LossSVRDual((LBF_DATA*)binaryfeatures.data, binaryfeatures.rows, binaryfeatures.cols,
		deltashapesArray, deltashapes.size(), deltashapes[0].size(),
		(LBF_DATA*)curW.data, curW.rows, curW.cols);

	delete[] deltashapesArray;

	//GetGeometricTransform(binaryfeatures, deltashapesMat, W_liblinear);
	//cv::transpose(W_liblinear, W_liblinear);

	//toc;
	//W = W_liblinear;

	// Predict the location of lanmarks using current regression matrix

	//cv::mat을 이용해서 매트릭스 연산 할것
	//deltashapes_bar = binaryfeatures*W_liblinear( matrix 연산)
	//cv::Mat deltashapes_bar(W_liblinear.cols, dbsize * m_Params.augnumber, LBF_MAT_TYPE);
	//cv::Mat deltashapes_bar(binaryfeatures.rows, binaryfeatures.cols, LBF_MAT_TYPE);
	cv::Mat deltashapes_bar = binaryfeatures * curW;

	std::vector< Shape > predshapes(binaryfeatures.rows);// concatenate 2 - D coordinates into a vector(N X(2 * L))
	//predshapes.resize(m_Params.meanshape.size());
	//predshapes = zeros([size(m_Params.meanshape) size(binaryfeatures, 1)]);  

	for (int i = 0; i < dbsize*(m_Params.augnumber); i++)
	{
		int k = std::floor(i / (m_Params.augnumber));
		int s = i % (m_Params.augnumber);
		std::vector<Shape> shapes_stage(m_vData[k].intermediateShapes[stage].size());
		for (int j = 0; j < shapes_stage.size(); j++)
			shapes_stage[j] = m_vData[k].intermediateShapes[stage][j].clone();

		Shape shape_stage = shapes_stage[s].clone();

		Shape deltashapes_bar_xy; Shape delta_shape_interm_coord;
		deltashapes_bar_xy.resize(deltashapes_bar.cols / 2);
		for (int j = 0; j < deltashapes_bar_xy.size(); j++)
		{
			deltashapes_bar_xy.data()[j].x = deltashapes_bar.at<LBF_DATA>(i, j);
			deltashapes_bar_xy.data()[j].y = deltashapes_bar.at<LBF_DATA>(i, j + deltashapes_bar_xy.size());
		}

		// transform above delta shape into the coordinate of current intermmediate shape
		// delta_shape_interm_coord = [deltashapes_bar_x(i, :)', deltashapes_bar_y(i, :)'];

		cv::Mat deltashapes_bar_xyMat;
		deltashapes_bar_xy.ToTransMat(deltashapes_bar_xyMat);
		DoGeometricTransform(m_vData[k].meanShape2tf[s], deltashapes_bar_xyMat, deltashapes_bar_xyMat);
		delta_shape_interm_coord.FromTransMat(deltashapes_bar_xyMat);

		// derive the delta shape in the coordinate system of meanshape

		for (int j = 0; j < delta_shape_interm_coord.size(); j++)
		{
			delta_shape_interm_coord.data()[j].x *= m_vData[k].intermediateBboxes[stage][s].width;
			delta_shape_interm_coord.data()[j].y *= m_vData[k].intermediateBboxes[stage][s].height;
		}

		Shape shape_newstage(shape_stage.size());
		for (int j = 0; j < shape_newstage.size(); j++)
		{
			shape_newstage.data()[j] = shape_stage.data()[j] + delta_shape_interm_coord.data()[j];
		}
		predshapes[i] = shape_newstage.clone();

		m_vData[k].intermediateShapes[stage + 1][s] = shape_newstage.clone();

		// update transformation of current intermediate shape to meanshape
		m_vData[k].intermediateBboxes[stage + 1][s] = m_vData[k].intermediateShapes[stage + 1][s].GetBbox(m_vData[k].originalWidth, m_vData[k].originalHeight);
		Shape meanshape_resize = m_Params.meanShape.ResetShape(m_vData[k].intermediateBboxes[stage + 1][s]);
		Shape shape_residual(m_vData[k].gtShape.size());
		for (int j = 0; j < m_vData[k].gtShape.size(); j++)
		{
			shape_residual.data()[j].x = (m_vData[k].gtShape.data()[j].x - shape_newstage.data()[j].x) / m_vData[k].intermediateBboxes[stage + 1][s].width;
			shape_residual.data()[j].y = (m_vData[k].gtShape.data()[j].y - shape_newstage.data()[j].y) / m_vData[k].intermediateBboxes[stage + 1][s].height;
		}


		Shape intermLocalShape(m_vData[k].intermediateShapes[stage + 1][s].size());
		Shape meanLocalShape(meanshape_resize.size());
		LBF_POINT intermPoint = m_vData[k].intermediateShapes[stage + 1][s].GetMeanPoint();
		LBF_POINT meanPoint = meanshape_resize.GetMeanPoint();
		for (int j = 0; j < intermLocalShape.size(); j++)
		{
			intermLocalShape.data()[j] = m_vData[k].intermediateShapes[stage + 1][s].data()[j] - intermPoint;
			meanLocalShape.data()[j] = meanshape_resize.data()[j] - meanPoint;
		}

		cv::Mat intermMat, meanResiedMat;
		intermLocalShape.ToTransMat(intermMat);
		meanLocalShape.ToTransMat(meanResiedMat);
		GetGeometricTransform(intermMat, meanResiedMat, m_vData[k].tf2meanShape[s]);
		GetGeometricTransform(meanResiedMat, intermMat, m_vData[k].meanShape2tf[s]);

		cv::Mat shape_residualMat;
		shape_residual.ToTransMat(shape_residualMat);
		DoGeometricTransform(m_vData[k].tf2meanShape[s], shape_residualMat, shape_residualMat);
		m_vData[k].shapeResidual[s].FromTransMat(shape_residualMat);

		/*drawshapes(m_vData{ k }.img_gray, [m_vData{ k }.shape_gt shape_stage shape_newstage]);
		hold off;
		drawnow;
		w = waitforbuttonpress;*/
	}

	std::vector<LBF_DATA> error_per_image;
	ComputeError(gtshapes, predshapes, error_per_image);
	LBF_DATA MRSE = 0.0;
	for (int i = 0; i < error_per_image.size(); i++)
		MRSE += error_per_image[i];
	MRSE = 100 * (MRSE / error_per_image.size());
	std::cout << "Mean Root Square Error for " << error_per_image.size() << " Test Samples: " << MRSE << std::endl;

	return;
}

void LBF::GlobalPrediction(const cv::Mat& binaryfeatures, int stage)
{
	cv::Mat& curW = m_Model.Ws[stage];

	// organize the groundtruth shape
	int dbsize = m_vData.size();
	std::vector< Shape > gtshapes(dbsize*m_Params.augnumber);

	// concatenate 2 - D coordinates into a vector(N X(2 * L))
	for (int i = 0; i < gtshapes.size(); i++)
		gtshapes[i].resize(m_Params.meanShape.size());

	std::vector<LBF_DATA> dist_pupils(dbsize*m_Params.augnumber);

	for (int i = 0; i < dbsize*m_Params.augnumber; i++)
	{
		int k = std::floor(i / (m_Params.augnumber));
		int s = i % (m_Params.augnumber);

		Shape shape_gt = m_vData[k].gtShape.clone();
		gtshapes[i] = shape_gt.clone();

		// left eye : 37 - 42
		// right eye : 43 - 48
		if (shape_gt.size() == 68)
		{
			LBF_POINT shape_gt_mean1, shape_gt_mean2;
			std::vector<int> used1, used2;

			GetUsedPoints(36, 41, 1, used1);
			shape_gt_mean1 = shape_gt.GetMeanUsedPoint(used1);

			GetUsedPoints(42, 47, 1, used2);
			shape_gt_mean2 = shape_gt.GetMeanUsedPoint(used2);

			GetDistance(shape_gt_mean1, shape_gt_mean2, dist_pupils[i]);
		}
		else if (shape_gt.size() == 51)
			GetDistance(shape_gt.data()[19], shape_gt.data()[28], dist_pupils[i]);

		else if (shape_gt.size() == 29)
		{
			LBF_POINT shape_gt_mean1, shape_gt_mean2;
			std::vector<int> used1, used2;

			GetUsedPoints(8, 16, 2, used1);
			shape_gt_mean1 = shape_gt.GetMeanUsedPoint(used1);

			GetUsedPoints(9, 17, 2, used2);
			shape_gt_mean2 = shape_gt.GetMeanUsedPoint(used2);

			GetDistance(shape_gt_mean1, shape_gt_mean2, dist_pupils[i]);
		}
	}

	// Predict the location of lanmarks using current regression matrix
	cv::Mat deltasahpes_bar = binaryfeatures * curW;

	std::vector< Shape > predshapes(dbsize * m_Params.augnumber);// concatenate 2 - D coordinates into a vector(N X(2 * L))

	for (int i = 0; i < dbsize*m_Params.augnumber; i++)
	{
		int k = std::floor(i / (m_Params.augnumber));
		int s = i % (m_Params.augnumber);

		std::vector<Shape> shapes_stage(m_vData[k].intermediateShapes[stage].size());
		for (int j = 0; j < shapes_stage.size(); j++)
			shapes_stage[j] = m_vData[k].intermediateShapes[stage][j].clone();
		Shape shape_stage = shapes_stage[s].clone();

		Shape deltashapes_bar_xy(deltasahpes_bar.cols / 2);
		Shape delta_shape_intermmed_coord, delta_shape_meanshape_coord;
		for (int j = 0; j < deltashapes_bar_xy.size(); j++)
		{
			deltashapes_bar_xy.data()[j].x = deltasahpes_bar.at<LBF_DATA>(i, j);
			deltashapes_bar_xy.data()[j].y = deltasahpes_bar.at<LBF_DATA>(i, j + deltashapes_bar_xy.size());
		}
		// transform above delta shape into the coordinate of current intermmediate shape
		// delta_shape_intermmed_coord = deltashapes_bar_xy;

		cv::Mat deltashapes_bar_xyMat;
		deltashapes_bar_xy.ToTransMat(deltashapes_bar_xyMat);
		DoGeometricTransform(m_vData[k].meanShape2tf[s], deltashapes_bar_xyMat, deltashapes_bar_xyMat);
		delta_shape_intermmed_coord.FromTransMat(deltashapes_bar_xyMat);

		delta_shape_meanshape_coord.resize(delta_shape_intermmed_coord.size());
		for (int j = 0; j < delta_shape_intermmed_coord.size(); j++)
		{
			delta_shape_meanshape_coord.data()[j].x = delta_shape_intermmed_coord.data()[j].x * m_vData[k].intermediateBboxes[stage][s].width;
			delta_shape_meanshape_coord.data()[j].y = delta_shape_intermmed_coord.data()[j].y * m_vData[k].intermediateBboxes[stage][s].height;
		}

		Shape shape_newstage(shape_stage.size());
		for (int j = 0; j < shape_newstage.size(); j++)
		{
			shape_newstage.data()[j] = shape_stage.data()[j] + delta_shape_meanshape_coord.data()[j];
		}

		//predshapes(:, : , i) = reshape(shape_newstage(:), size(m_Params.meanshape));
		//왜 똑같은 사이즈를 똑같은 사이즈로 reshape하는지 모르겠음
		predshapes[i] = shape_newstage.clone();

		m_vData[k].intermediateShapes[stage + 1][s] = shape_newstage.clone();
		m_vData[k].intermediateBboxes[stage + 1][s] = m_vData[k].intermediateBboxes[stage][s];	// getbbox(shape_newstage);

		// update transformation of current intermediate shape to meanshape
		Shape meanshape_resize = m_Params.meanShape.ResetShape(m_vData[k].intermediateBboxes[stage + 1][s]);


		Shape intermLocalShape(m_vData[k].intermediateShapes[stage + 1][s].size());
		Shape meanLocalShape(meanshape_resize.size());
		LBF_POINT intermPoint = m_vData[k].intermediateShapes[stage + 1][s].GetMeanPoint();
		LBF_POINT meanPoint = meanshape_resize.GetMeanPoint();
		for (int j = 0; j < intermLocalShape.size(); j++)
		{
			intermLocalShape.data()[j] = m_vData[k].intermediateShapes[stage + 1][s].data()[j] - intermPoint;
			meanLocalShape.data()[j] = meanshape_resize.data()[j] - meanPoint;
		}

		cv::Mat intermMat, meanResizedMat;
		intermLocalShape.ToTransMat(intermMat);
		meanLocalShape.ToTransMat(meanResizedMat);
		GetGeometricTransform(intermMat, meanResizedMat, m_vData[k].tf2meanShape[s]);
		GetGeometricTransform(meanResizedMat, intermMat, m_vData[k].meanShape2tf[s]);

		/*
		if stage >= m_Params.max_numstage
		%[m_vData{ k }.shape_gt m_vData{ k }.intermediate_shapes{ 1 }(:, : , s) shape_newstage]
		drawshapes(m_vData{ k }.img_gray, [m_vData{ k }.shape_gt m_vData{ k }.intermediate_shapes{ 1 }(:, : , s) shape_newstage]);
		hold off;
		drawnow;
		error_per_image = ComputeError(gtshapes(:, : , i), predshapes(:, : , i))
		w = waitforbuttonpress;
		end
		*/
	}

	std::vector<LBF_DATA> error_per_image;
	ComputeError(gtshapes, predshapes, error_per_image);
	LBF_DATA MRSE = 0.0;
	for (int i = 0; i < error_per_image.size(); i++)
		MRSE += error_per_image[i];
	MRSE = 100 * (MRSE / error_per_image.size());
	std::cout << "Mean Root Square Error for " << error_per_image.size() << " Test Samples: " << MRSE << std::endl;
	return;
}
