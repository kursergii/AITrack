#pragma once

#include <string>
#include <opencv2/opencv.hpp>

struct CameraMotion {
    cv::Point2f direction;      // Translation vector (scene motion in frame)
    std::string directionName;
    float magnitude;
};

// Compute camera/scene motion using optical flow
CameraMotion computeCameraMotion(const cv::Mat& prevGray, const cv::Mat& currGray);
