#include "camera_motion.hpp"
#include <cmath>
#include <vector>

CameraMotion computeCameraMotion(const cv::Mat& prevGray, const cv::Mat& currGray) {
    CameraMotion motion = {{0, 0}, "STILL", 0};

    // Find good features to track in previous frame
    std::vector<cv::Point2f> prevPoints;
    cv::goodFeaturesToTrack(prevGray, prevPoints, 300, 0.01, 10);

    if (prevPoints.size() < 10) return motion;

    // Track points using optical flow with larger window for fast motion
    std::vector<cv::Point2f> currPoints;
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(prevGray, currGray, prevPoints, currPoints, status, err,
                             cv::Size(21, 21), 3);  // Larger window, more pyramid levels

    // Compute average motion from valid tracked points
    // Use higher error tolerance for fast motion
    cv::Point2f totalMotion(0, 0);
    int validCount = 0;

    for (size_t i = 0; i < status.size(); i++) {
        if (status[i] && err[i] < 30.0f) {  // Increased error tolerance
            totalMotion += (currPoints[i] - prevPoints[i]);
            validCount++;
        }
    }

    if (validCount < 5) return motion;  // Lower minimum for sparse scenes

    // Average scene motion (how scene points move in frame)
    motion.direction = totalMotion / static_cast<float>(validCount);
    motion.magnitude = std::sqrt(motion.direction.x * motion.direction.x +
                                  motion.direction.y * motion.direction.y);

    // Determine direction name (lower threshold to detect subtle camera motion)
    if (motion.magnitude < 0.8f) {
        motion.directionName = "STILL";
    } else {
        float angle = std::atan2(-motion.direction.y, motion.direction.x) * 180.0f / static_cast<float>(CV_PI);

        if (angle >= -22.5f && angle < 22.5f) {
            motion.directionName = "RIGHT";
        } else if (angle >= 22.5f && angle < 67.5f) {
            motion.directionName = "UP-RIGHT";
        } else if (angle >= 67.5f && angle < 112.5f) {
            motion.directionName = "UP";
        } else if (angle >= 112.5f && angle < 157.5f) {
            motion.directionName = "UP-LEFT";
        } else if (angle >= 157.5f || angle < -157.5f) {
            motion.directionName = "LEFT";
        } else if (angle >= -157.5f && angle < -112.5f) {
            motion.directionName = "DOWN-LEFT";
        } else if (angle >= -112.5f && angle < -67.5f) {
            motion.directionName = "DOWN";
        } else {
            motion.directionName = "DOWN-RIGHT";
        }
    }

    return motion;
}
