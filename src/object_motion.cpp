/**
 * @file object_motion.cpp
 * @brief Object motion computation with camera motion compensation.
 *
 * Computes object motion in world coordinates by subtracting camera motion
 * from observed frame motion.
 */

#include "object_motion.hpp"
#include <cmath>

/**
 * @brief Compute object motion with camera compensation.
 *
 * Frame motion is the raw displacement of object center between frames.
 * Environment motion = frame motion - camera motion, giving the object's
 * true motion independent of camera panning.
 */
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

    // Determine direction
    if (motion.magnitude < ObjectMotionParams::STILL_THRESHOLD) {
        motion.directionName = "STILL";
    } else {
        // 8-direction quantization: each sector is 45° (360°/8), boundaries at ±22.5°
        constexpr float SECTOR_HALF = 22.5f;
        float angle = std::atan2(-motion.envMotion.y, motion.envMotion.x) * 180.0f / static_cast<float>(CV_PI);

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
