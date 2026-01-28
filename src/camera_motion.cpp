/**
 * @file camera_motion.cpp
 * @brief Camera motion estimation using sparse optical flow.
 *
 * Estimates global scene/camera motion by tracking feature points between
 * consecutive frames using Lucas-Kanade optical flow.
 */

#include "camera_motion.hpp"
#include <cmath>
#include <vector>

/**
 * @brief Compute camera/scene motion between frames.
 *
 * Algorithm:
 * 1. Find good features to track in previous frame (Shi-Tomasi corners)
 * 2. Track features using pyramidal Lucas-Kanade optical flow
 * 3. Average valid point displacements to get global motion
 * 4. Quantize direction to 8 compass directions
 */
CameraMotion computeCameraMotion(const cv::Mat& prevGray, const cv::Mat& currGray) {
    CameraMotion motion = {{0, 0}, "STILL", 0};

    // Find good features to track in previous frame
    std::vector<cv::Point2f> prevPoints;
    cv::goodFeaturesToTrack(prevGray, prevPoints, OpticalFlowParams::MAX_FEATURES,
                            OpticalFlowParams::QUALITY_LEVEL, OpticalFlowParams::MIN_DISTANCE);

    if (prevPoints.size() < static_cast<size_t>(OpticalFlowParams::MIN_FEATURES_TO_COMPUTE))
        return motion;

    // Track points using optical flow with larger window for fast motion
    std::vector<cv::Point2f> currPoints;
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(prevGray, currGray, prevPoints, currPoints, status, err,
                             cv::Size(OpticalFlowParams::WINDOW_SIZE, OpticalFlowParams::WINDOW_SIZE),
                             OpticalFlowParams::PYRAMID_LEVELS);

    // Compute average motion from valid tracked points
    cv::Point2f totalMotion(0, 0);
    int validCount = 0;

    for (size_t i = 0; i < status.size(); i++) {
        if (status[i] && err[i] < OpticalFlowParams::MAX_ERROR) {
            totalMotion += (currPoints[i] - prevPoints[i]);
            validCount++;
        }
    }

    if (validCount < OpticalFlowParams::MIN_VALID_POINTS) return motion;

    // Average scene motion (how scene points move in frame)
    motion.direction = totalMotion / static_cast<float>(validCount);
    motion.magnitude = std::sqrt(motion.direction.x * motion.direction.x +
                                  motion.direction.y * motion.direction.y);

    // Determine direction name (lower threshold to detect subtle camera motion)
    if (motion.magnitude < OpticalFlowParams::STILL_THRESHOLD) {
        motion.directionName = "STILL";
    } else {
        // 8-direction quantization: each sector is 45° (360°/8), boundaries at ±22.5°
        constexpr float SECTOR_HALF = 22.5f;  // Half of 45° sector
        float angle = std::atan2(-motion.direction.y, motion.direction.x) * 180.0f / static_cast<float>(CV_PI);

        if (angle >= -SECTOR_HALF && angle < SECTOR_HALF) {
            motion.directionName = "RIGHT";
        } else if (angle >= SECTOR_HALF && angle < 3 * SECTOR_HALF) {
            motion.directionName = "UP-RIGHT";
        } else if (angle >= 3 * SECTOR_HALF && angle < 5 * SECTOR_HALF) {
            motion.directionName = "UP";
        } else if (angle >= 5 * SECTOR_HALF && angle < 7 * SECTOR_HALF) {
            motion.directionName = "UP-LEFT";
        } else if (angle >= 7 * SECTOR_HALF || angle < -7 * SECTOR_HALF) {
            motion.directionName = "LEFT";
        } else if (angle >= -7 * SECTOR_HALF && angle < -5 * SECTOR_HALF) {
            motion.directionName = "DOWN-LEFT";
        } else if (angle >= -5 * SECTOR_HALF && angle < -3 * SECTOR_HALF) {
            motion.directionName = "DOWN";
        } else {
            motion.directionName = "DOWN-RIGHT";
        }
    }

    return motion;
}
