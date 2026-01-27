#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

// Draw predicted trajectory
void drawPrediction(cv::Mat& frame, const cv::Point2f& currentCenter,
                    const std::vector<cv::Point2f>& predictedPath, float confidence);

// Draw motion arrow indicator
void drawMotionIndicator(cv::Mat& frame, const cv::Point2f& direction, float magnitude,
                         cv::Point center, cv::Scalar color, float threshold = 1.5f);
