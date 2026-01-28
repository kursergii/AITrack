/**
 * @file object_motion.hpp
 * @brief Object motion computation with camera motion compensation.
 *
 * Computes object motion in both frame coordinates and world/environment
 * coordinates by subtracting the camera motion from the observed frame motion.
 */

#pragma once

#include <string>
#include <opencv2/opencv.hpp>
#include "camera_motion.hpp"

/// Object motion detection parameters
namespace ObjectMotionParams {
    constexpr float STILL_THRESHOLD = 1.0f;  ///< Minimum motion to not be "STILL" (pixels/frame)
}

/**
 * @struct ObjectMotion
 * @brief Represents object motion in both frame and environment coordinates.
 */
struct ObjectMotion {
    cv::Point2f frameMotion;     ///< Motion as observed in frame (affected by camera motion)
    cv::Point2f envMotion;       ///< Motion relative to environment (camera compensated)
    std::string directionName;   ///< Human-readable direction of envMotion
    float magnitude;             ///< Magnitude of envMotion in pixels/frame
};

/**
 * @brief Compute object motion with camera motion compensation.
 * @param prevBbox Previous bounding box
 * @param currBbox Current bounding box
 * @param cameraMotion Estimated camera motion to compensate for
 * @return ObjectMotion with frame motion, environment motion, direction, and magnitude
 *
 * Environment motion = frame motion - camera motion, giving the object's true motion
 * in world coordinates independent of camera panning.
 */
ObjectMotion computeObjectMotion(const cv::Rect& prevBbox, const cv::Rect& currBbox,
                                  const CameraMotion& cameraMotion);
