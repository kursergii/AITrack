#pragma once

#include <string>
#include <opencv2/opencv.hpp>
#include "camera_motion.hpp"

struct ObjectMotion {
    cv::Point2f frameMotion;     // Motion in frame coordinates
    cv::Point2f envMotion;       // Motion relative to environment
    std::string directionName;
    float magnitude;
};

// Compute object motion relative to environment (compensated for camera motion)
ObjectMotion computeObjectMotion(const cv::Rect& prevBbox, const cv::Rect& currBbox,
                                  const CameraMotion& cameraMotion);
