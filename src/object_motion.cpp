#include "object_motion.hpp"
#include <cmath>

ObjectMotion computeObjectMotion(const cv::Rect& prevBbox, const cv::Rect& currBbox,
                                  const CameraMotion& cameraMotion) {
    ObjectMotion motion = {{0, 0}, {0, 0}, "STILL", 0};

    // Object center movement in frame
    cv::Point2f prevCenter(prevBbox.x + prevBbox.width / 2.0f,
                           prevBbox.y + prevBbox.height / 2.0f);
    cv::Point2f currCenter(currBbox.x + currBbox.width / 2.0f,
                           currBbox.y + currBbox.height / 2.0f);

    motion.frameMotion = currCenter - prevCenter;

    // Compensate for camera motion to get object motion in environment
    // cameraMotion.direction = how scene points move in frame (optical flow)
    // If camera pans RIGHT, scene moves LEFT in frame (negative x direction)
    // If object is stationary in env, it moves like the scene
    // So: envMotion = frameMotion - sceneMotion (to cancel out camera movement)
    motion.envMotion = motion.frameMotion - cameraMotion.direction;

    motion.magnitude = std::sqrt(motion.envMotion.x * motion.envMotion.x +
                                  motion.envMotion.y * motion.envMotion.y);

    // Determine direction (lower threshold to detect small movements)
    const float threshold = 1.0f;
    if (motion.magnitude < threshold) {
        motion.directionName = "STILL";
    } else {
        float angle = std::atan2(-motion.envMotion.y, motion.envMotion.x) * 180.0f / static_cast<float>(CV_PI);

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
