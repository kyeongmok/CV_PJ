#include <fstream>
#include <iomanip>

#include "Liblinear.h"
#include "LBF.h"
#include "Shape.h"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"

class CV_PJ_Face
{
public:
	inline void GetImglistVector(std::vector<std::string>& trainImgPathList){ trainImgPathList = m_TrainImgPathList; }

	CV_PJ_Face(std::string _dataDirPath)
	{
		m_DataDirPath = _dataDirPath;
	}

	void CV_PJ_LoadModel()
	{
		m_LBFModel.SetLBFParams(8, 4, 0, 5, 0.4);

		/* Load FD Model */
		//std::cout << "loading FaceDetector Model" << std::endl;
		m_FaceDetector.load(m_DataDirPath + "//haarcascade_frontalface_default.xml");

		/* Load LBF Model */
		std::string initShapePath = m_DataDirPath + "//InitialShape_68.pts";

		m_LBFModel.InitFrame(initShapePath);
		m_LBFModel.LoadModel(m_DataDirPath + "//RandomForest_FD_8_4_0_5.txt", m_DataDirPath + "//Ws_FD_8_4_0_5.xml");
	}

	void CV_PJ_Detect(const cv::Mat& inputImg, cv::Rect& faceBbox, std::vector<cv::Point2d>& landmarkPoints, std::vector<cv::Point2d>& landmarkRegularized, float regularizeScale = 1.0)
	{
		//preprocessing
		cv::Mat curGrayFrame = inputImg.clone();
		if (curGrayFrame.channels() != 1)
		{
			cv::cvtColor(curGrayFrame, curGrayFrame, CV_BGR2GRAY);
		}

		//face detection
		std::vector<Bbox> vFace;
		m_FaceDetector.detectMultiScale(curGrayFrame, vFace, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));

		if (!vFace.empty())
		{
			faceBbox = vFace[0];
		}

		//face alignment
		Shape predShape;
		m_LBFModel.Prediction(curGrayFrame, vFace[0], predShape);
		landmarkPoints = predShape.data();
		landmarkRegularized = predShape.RegularizeShape(vFace[0], regularizeScale).data();
	}

	float CV_PJ_Scoring(std::vector<cv::Point2d> facePts) {
    double lengthBtneye = cv::norm(facePts[39] - facePts[42]);
    double rightEye = cv::norm(facePts[42] - facePts[45]);
    double leftEye = cv::norm(facePts[39] - facePts[36]);
    double noseLengthV = cv::norm(facePts[27] - facePts[33]);
    double noseLengthW = cv::norm(facePts[31] - facePts[35]);
    double lipLengthV = cv::norm(facePts[51] - facePts[57]);
    double lipLengthW = cv::norm(facePts[48] - facePts[54]);
    double faceWidth = cv::norm(facePts[0] - facePts[16]);
    cv::Point2d centerOfEyeBrow = cv::Point2d((facePts[21].x + facePts[22].x) / 2.0f, (facePts[21].y + facePts[22].y) / 2.0f);
    double faceHeight = cv::norm(centerOfEyeBrow - facePts[8]);
    double noseToJaw = cv::norm(facePts[33] - facePts[8]);
    double upLip = cv::norm(facePts[51] - facePts[62]);
    double lowLip = cv::norm(facePts[62] - facePts[57]);
    double bow = cv::norm(facePts[17] - facePts[21]);

    double ratioScore1 = 15-(std::abs(1.43-lengthBtneye/rightEye)/1.43*15);
    double ratioScore2 = 15-((std::abs((noseLengthV/noseLengthW)-1.05))/1.05*15);
    double ratioScore3 = 15-((std::abs(0.52-(noseLengthV/noseToJaw))/0.52)*15);
    double ratioScore4 = 4-((std::abs(2.37-(lipLengthW/lipLengthV)))/2.37*4);
    double ratioScore5 = 11-(std::abs(1.76-(lowLip/upLip))/1.76*11);
    double ratioScore6 = 15-(std::abs(1.58-(noseLengthW/rightEye))/1.58*15);
    double ratioScore7 = 15-((std::abs(0.83-(faceHeight/faceWidth))/0.83)*15);
    double ratio = ratioScore7/15;

    double ratioScore1b = 15-(std::abs(1.43-lengthBtneye/leftEye)/1.43*15);
    double ratioScore6b = 15-(std::abs(1.58-noseLengthW/leftEye)/1.58*15);

    double diff = 10-(std::abs((ratioScore1+ratioScore6) - (ratioScore1b+ratioScore6b))*10);

    double sum = (ratioScore1+ratioScore2+ratioScore3+ratioScore4+ratioScore5+ratioScore6+ratioScore7)+(ratioScore1b+ratioScore2+ratioScore3+ratioScore4+ratioScore5+ratioScore6b+ratioScore7);
    double avg = ((sum/2)+diff)*ratio;
    return avg;
  }

	float CV_PJ_Scoring2(std::vector<cv::Point2d> facePts) {
    cv::Point2d lipCenter = getCenterPoint(facePts[62], facePts[66]);
    cv::Point2d eyeCenter = getCenterPoint(getCenterPoint(facePts[36], facePts[39]), getCenterPoint(facePts[42], facePts[45]));
    cv::Point2d centerOfEyeBrow = getCenterPoint(facePts[21], facePts[22]);
    double eyeToLip = cv::norm(eyeCenter - lipCenter);
    double LipToChin = cv::norm(lipCenter - facePts[8]);

    double noseToChin = cv::norm(facePts[33] - facePts[8]);
    double noseLengthV = cv::norm(facePts[27] - facePts[33]);
    double noseLengthW = cv::norm(facePts[31] - facePts[35]);
    double noseToLip = cv::norm(facePts[33] - lipCenter);
    // vertical ratio of face
    double vRatio1 = eyeToLip / LipToChin;  // 1.618
    double vRatio2 = noseToChin / noseLengthV; // 1.618
    double vRatio3 = noseLengthV / noseToLip; // 1.618
    double vRatio4 = LipToChin / noseToLip; // 1.618
    double vMean = (vRatio1 + vRatio2 + vRatio3 + vRatio4) / 4.0f; // 1.618
    // facial third method ratio 
    double thirdRatio1 = cv::norm(centerOfEyeBrow - facePts[33]) / noseToChin; // 1
    double thirdRatio2 = cv::norm(facePts[51] - facePts[8]) / cv::norm(facePts[33] - facePts[51]); // should be 2

    // 70점 만점. 
    double score = getScoreOfRatio(vRatio1) + getScoreOfRatio(vRatio2) + getScoreOfRatio(vRatio3) + getScoreOfRatio(vRatio4) + getScoreOfRatio(vMean) + getScoreOfRatio(thirdRatio1, 1.0f) + getScoreOfRatio(thirdRatio2, 2);

    return score;
  }

	void ReadTrainList()
	{
		/* train set load */
		std::string imgPathListFile = m_DataDirPath + "//scoreDataset//Path_Trainset.txt";
		std::ifstream inFile;
		inFile.open(imgPathListFile.c_str());
		if (inFile.is_open())
		{
			while (!inFile.eof())
			{
				std::string tmpStr;
				std::getline(inFile, tmpStr);
				if (tmpStr != "")
					m_TrainImgPathList.push_back(tmpStr);
			}
			inFile.close();
		}
		else
		{
			std::cerr << "fail : load image path list file" << std::endl;
			return;
		}
	}

	void TrainScoreModel(bool willSave)
	{
		/* get train matrix */
		int numSamples, inputNumFeatures, targetNumFeatures;
		numSamples = m_TrainImgPathList.size();
		inputNumFeatures = 68 * 2;	// # of lmarks(x and y coord)
		targetNumFeatures = 1;	// # of output score
		double *pInputMatrix, *pTargetMatrix, *pRegressionResult;
		pInputMatrix = new double[numSamples * inputNumFeatures];
		pTargetMatrix = new double[numSamples * targetNumFeatures];
		pRegressionResult = new double[inputNumFeatures * targetNumFeatures];

		assert(numSamples != 0);

		for (int i = 0; i < numSamples; i++)
		{
			/* parsing to get the score vector */
			std::string curImgPath = m_TrainImgPathList[i];
			std::string::size_type pos = 0;
			int startPos, endPos;
			startPos = endPos = 0;
			std::string startFind = "\\";
			std::string endFind = "_";
			while ((pos = curImgPath.find(startFind, pos)) != std::string::npos)
			{
				pos += startFind.size();
				startPos = pos;
			}
			pos = 0;
			while ((pos = curImgPath.find(endFind, pos)) != std::string::npos)
			{
				pos += endFind.size();
				endPos = pos;
			}
			std::string scoreStr = curImgPath.substr(startPos, endPos - startPos - 1);
			double curScore = std::stod(scoreStr);
			
			/* face alignment */
			cv::Mat curGrayImg = cv::imread(m_TrainImgPathList[i].c_str(), CV_LOAD_IMAGE_GRAYSCALE);
			
//			cv::resize(curGrayImg, curGrayImg, cv::Size(640, 480));

			cv::Rect detectedFace;
			std::vector<cv::Point2d> lmarks, lmarksRegularized;
			CV_PJ_Detect(curGrayImg, detectedFace, lmarks, lmarksRegularized, 1.0);

			//tmp drawing
// 			for (int i = 0; i < lmarks.size(); i++)
// 			{
// 				cv::circle(curGrayImg, lmarks[i], 2, cv::Scalar(0, 0, 255), -1);
// 			}
// 			cv::imshow("test", curGrayImg);
// 			cv::waitKey();


			if (lmarksRegularized.size() != inputNumFeatures / 2)
			{
				std::cerr << "num landmarks error\n";
				return;
			}
			
			for (int j = 0; j < inputNumFeatures / 2; j++)
			{
				pInputMatrix[i * inputNumFeatures + 2 * j] = lmarksRegularized[j].x;
				pInputMatrix[i * inputNumFeatures + 2 * j + 1] = lmarksRegularized[j].y;
			}
			pTargetMatrix[i * targetNumFeatures] = curScore;
		}

		/* Train */
		L2RegularL2LossSVRDual(pInputMatrix, numSamples, inputNumFeatures, pTargetMatrix, numSamples, targetNumFeatures, pRegressionResult, inputNumFeatures, targetNumFeatures);
		cv::Mat scoreModel = cv::Mat(inputNumFeatures, targetNumFeatures, CV_64F, pRegressionResult);

		/* Save Model */
		if (willSave)
		{
			std::string modelPath = m_DataDirPath + "\\ScoreModel.xml";
			cv::FileStorage fs(modelPath, cv::FileStorage::WRITE);
			fs << "ScoreModel" << scoreModel;
			fs.release();
		}
		else
		{
			m_ScoreModel.release();
			m_ScoreModel = scoreModel.clone();
		}

		delete[] pInputMatrix;
		delete[] pTargetMatrix;
		scoreModel.release();
	}

	void LoadScoreModel()
	{
		std::string modelPath = m_DataDirPath + "\\ScoreModel.xml";
		cv::FileStorage fs(modelPath, cv::FileStorage::READ);
		fs["ScoreModel"] >> m_ScoreModel;
		fs.release();
	}

	double PredictScore(const std::vector<cv::Point2d>& lmarkRegularized)
	{
		if (m_ScoreModel.empty())
		{
			std::cerr << "Score Model has not been loaded\n";
			return -1;
		}
		cv::Mat inputMat(1, lmarkRegularized.size() * 2, CV_64F);
		for (int i = 0; i < lmarkRegularized.size(); i++)
		{
			inputMat.at<double>(0, i * 2) = lmarkRegularized.data()[i].x;
			inputMat.at<double>(0, i * 2 + 1) = lmarkRegularized.data()[i].y;
		}
		cv::Mat resultScore = inputMat * m_ScoreModel;

		return resultScore.at<double>(0, 0);
	}

private:
	LBF m_LBFModel;
	cv::CascadeClassifier m_FaceDetector;
	std::string m_DataDirPath;
	cv::Mat m_ScoreModel;
	std::vector<std::string> m_TrainImgPathList;

	cv::Point2d getCenterPoint(cv::Point2d pt1, cv::Point2d pt2) 
  {
    return cv::Point2d((pt1.x + pt2.x) / 2.0f, (pt1.y + pt2.y) / 2.0f);
  }

	// 10점 만점.
	double getScoreOfRatio(double ratio, double targetRatio = 1.618f)
  {
    return 10 - (std::abs(ratio - targetRatio) / targetRatio) * 10;
  }
};

void CrossValidation()
{
	//Leave-one-out Cross-Validation
	std::string dataDirPath;
	dataDirPath = "..//..//Data";

	CV_PJ_Face cFace(dataDirPath);
	cv::Rect detectedFace;
	std::vector<cv::Point2d> detectedLandmarks, detectedRegularized;
	cFace.CV_PJ_LoadModel();

	cFace.ReadTrainList();
	std::vector<std::string> imgPathList;
	cFace.GetImglistVector(imgPathList);
	int numSample = imgPathList.size();
	assert(numSample != 0);

	std::ofstream outStream;
	std::string resultdataPath = "..//..//Data//crossval_resultData.txt";
	outStream.open(resultdataPath);
	if (!outStream.is_open())
		return;

	double avgError = 0;
	for (int i = 0; i < numSample; i++)
	{
		std::cout << "Train " << i << " of " << numSample << " case\n";
		std::string curTestImgPath = imgPathList[0];
		imgPathList.erase(imgPathList.begin());
		cFace.TrainScoreModel(false);

		/* parsing to get the score */
		std::string::size_type pos = 0;
		int startPos, endPos;
		startPos = endPos = 0;
		std::string startFind = "\\";
		std::string endFind = "_";
		while ((pos = curTestImgPath.find(startFind, pos)) != std::string::npos)
		{
			pos += startFind.size();
			startPos = pos;
		}
		pos = 0;
		while ((pos = curTestImgPath.find(endFind, pos)) != std::string::npos)
		{
			pos += endFind.size();
			endPos = pos;
		}
		std::string scoreStr = curTestImgPath.substr(startPos, endPos - startPos - 1);
		double groundTruthScore = std::stod(scoreStr);

		/* predict using the learned model */
		cv::Mat curTestImg = cv::imread(curTestImgPath.c_str());
		cFace.CV_PJ_Detect(curTestImg, detectedFace, detectedLandmarks, detectedRegularized, 1.0f);
		double predictedScore = cFace.PredictScore(detectedRegularized);

		/* calculate the error */
		double curError = std::abs(predictedScore - groundTruthScore);
		avgError += curError;
		std::cout << "curError : " << curError << "\n";

		/* save error data */
		outStream << curError << "\n";

		/* get back to the original vector */
		imgPathList.push_back(curTestImgPath);
	}
	avgError /= numSample;
	outStream << "Avg Error : " << avgError << "\n";
	outStream.close();

	std::cout << "leave-one-out cross-validation result\n average error : " << avgError << "\n";
}

int main()
//int main(int argc, char **argv)
{
//   cv::Mat testImg;
//   std::string dataDirPath, fileDirPath, fileName;
// // 	if (argc > 1) {
// // 		dataDirPath = argv[1];
// // 		fileDirPath = argv[2];
// // 		fileName = argv[3];
// //   } else {
// // 		dataDirPath = "..//Data";
// // 		fileDirPath = "..//Data";
// // 		fileName = "image_0030.png";
// //   }
// 	dataDirPath = "..//..//Data";
// 	fileDirPath = "..//..//Data";
// 	fileName = "image_0030.png";
// 
// 	testImg = cv::imread(fileDirPath + "//" + fileName);
// 
// 	if(testImg.empty())
// 	{
// 		std::cout << "image load failed" << "\n";
// 		return -1;
// 	}
// 
// 	CV_PJ_Face cFace(dataDirPath);
// 	cv::Rect detectedFace;
// 	std::vector<cv::Point2d> detectedLandmarks, detectedRegularized;
// 	cFace.CV_PJ_LoadModel();
// 	
// 	//cFace.ReadTrainList();
// 	//cFace.TrainScoreModel(true);
// 	cFace.LoadScoreModel();
// 
// 	cFace.CV_PJ_Detect(testImg, detectedFace, detectedLandmarks, detectedRegularized, 1.0f);
// 	double score = cFace.CV_PJ_Scoring(detectedLandmarks); 
// 	double score2 = cFace.CV_PJ_Scoring2(detectedLandmarks); 
// 	double score3 = cFace.PredictScore(detectedRegularized);
// 	std::cout << "face score = " << score << "\n";
// 	std::cout << "face score2 = " << score2 << "\n";
// 	std::cout << "predicted score = " << score3 << "\n";
// 	for (int i = 0; i < detectedLandmarks.size(); i++)
// 	{
// 		cv::circle(testImg, detectedLandmarks[i], 2, cv::Scalar(0, 0, 255), -1);
//     //cv::putText(testImg, std::to_string(i), detectedLandmarks[i], 1, 1, cv::Scalar(0,0,255));
// 	}
// 	std::string outputPath = fileDirPath + "//" + fileName;
// 	cv::imshow("test", testImg);
// 	cv::waitKey();
// 	//cv::imwrite(outputPath, testImg);
	
	CrossValidation();
	cv::waitKey();
	return 0;
}
