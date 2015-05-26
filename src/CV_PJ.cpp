#include "LBF.h"
#include "Shape.h"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"

class CV_PJ_Face
{
public:
	void CV_PJ_LoadModel()
	{
		m_LBFModel.SetLBFParams(8, 4, 0, 5, 0.4);

		/* Load FD Model */
		std::cout << "loading FaceDetector Model" << std::endl;
		m_FaceDetector.load("..\\Data\\haarcascade_frontalface_default.xml");

		/* Load LBF Model */
		std::string initShapePath = "..\\Data\\InitialShape_68.pts";
		m_LBFModel.InitFrame(initShapePath);
		m_LBFModel.LoadModel("..\\Data\\RandomForest_FD_8_4_0_5.txt", "..\\Data\\Ws_FD_8_4_0_5.xml");
	}

	void CV_PJ_Detect(const cv::Mat& inputImg, cv::Rect& faceBbox, std::vector<cv::Point2d>& landmarkPoints)
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
	}

private:
	LBF m_LBFModel;
	cv::CascadeClassifier m_FaceDetector;
};

int main()
{
	cv::Mat testImg = cv::imread("..\\Data\\image_0030.png");

	CV_PJ_Face cFace;
	cv::Rect detectedFace;
	std::vector<cv::Point2d> detectedLandmarks;
	cFace.CV_PJ_LoadModel();
	cFace.CV_PJ_Detect(testImg, detectedFace, detectedLandmarks);
	for (int i = 0; i < detectedLandmarks.size(); i++)
	{
		cv::circle(testImg, detectedLandmarks[i], 2, cv::Scalar(0, 0, 255), -1);
	}
	cv::imshow("resImg", testImg);
	cv::waitKey();

	return 0;
}