# CV_PJ

[dependancy]

- opencv : http://docs.opencv.org/doc/tutorials/introduction/linux_install/linux_install.html

[example program]

- main function in CV_PJ.cpp

[API usage]

- first, load models for prediction

    CV_PJ_Face::CV_PJ_LoadModel()
    
        (be sure that data file path is defined in this function)
        
- call prediction function
    
    CV_PJ_Face::CV_PJ_Detect(const cv::Mat& inputImg, cv::Rect& faceBbox, std::vector<cv::Point2d>& landmarkPoints)
    
      <input>
    
        inputImg : 1 or 3 channel input image to predict the facial landmarks
    
      <output>
    
        faceBbox : output rect object of face bounding box (integer coordinates)
    
        landmarkPoints : output vector of facial landmark points (double precision)
        
