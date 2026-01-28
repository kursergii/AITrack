/**
 * @file camera_motion.hpp
 * @brief Camera motion estimation using sparse optical flow.
 *
 * Detects camera/scene motion by tracking feature points between frames
 * using Lucas-Kanade optical flow. The estimated motion is used to compensate
 * for camera panning when computing object motion in world coordinates.
 */

#pragma once

#include <string>
#include <opencv2/opencv.hpp>

/// Optical flow algorithm parameters for camera motion estimation
namespace OpticalFlowParams {
    constexpr int MAX_FEATURES = 300;
    constexpr double QUALITY_LEVEL = 0.01;
    constexpr double MIN_DISTANCE = 10.0;
    constexpr int MIN_FEATURES_TO_COMPUTE = 10;
    constexpr int MIN_VALID_POINTS = 5;
    constexpr int WINDOW_SIZE = 21;
    constexpr int PYRAMID_LEVELS = 3;
    constexpr float MAX_ERROR = 30.0f;
    constexpr float STILL_THRESHOLD = 0.8f;
}

/**
 * @struct CameraMotion
 * @brief Represents estimated camera/scene motion between frames.
 */
struct CameraMotion {
    cv::Point2f direction;      ///< Translation vector (how scene points move in frame)
    std::string directionName;  ///< Human-readable direction (e.g., "LEFT", "UP-RIGHT")
    float magnitude;            ///< Motion magnitude in pixels/frame
};

/**
 * @brief Compute camera/scene motion using sparse optical flow.
 * @param prevGray Previous frame in grayscale
 * @param currGray Current frame in grayscale
 * @return CameraMotion struct with direction, magnitude, and direction name
 *
 * Uses goodFeaturesToTrack + calcOpticalFlowPyrLK to estimate global scene motion.
 * Returns STILL if insufficient features or motion below threshold.
 */
CameraMotion computeCameraMotion(const cv::Mat& prevGray, const cv::Mat& currGray);
