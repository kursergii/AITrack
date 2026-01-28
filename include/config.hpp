/**
 * @file config.hpp
 * @brief Centralized configuration for all tunable parameters.
 *
 * Provides default values and runtime configuration for tracker settings,
 * detection thresholds, class filtering, display options, and model paths.
 */

#pragma once

#include <string>
#include <vector>
#include <set>

/**
 * @struct Config
 * @brief Centralized configuration for all tunable parameters.
 */
struct Config {
    // Tracker settings
    int maxLostFrames = 30;           // Frames before track is considered lost
    float minIoU = 0.3f;              // Minimum IoU for detection-track matching
    float reinitThreshold = 0.7f;     // IoU threshold for tracker reinitialization
    int maxTrajectoryLength = 300;    // Max positions in trajectory history (~10s at 30fps)

    // Detection settings
    float detectionConfidence = 0.5f; // YOLO confidence threshold
    float nmsThreshold = 0.45f;       // Non-maximum suppression threshold
    int detectionInterval = 10;       // Run detection every N frames
    int inputSize = 640;              // YOLO input size

    // Class filtering (empty = track all classes)
    std::set<std::string> allowedClasses;  // Only track these classes (if not empty)
    std::set<std::string> blockedClasses;  // Never track these classes

    // Display settings
    int targetFps = 30;
    bool showTrajectoryMap = true;
    bool autoTrackDetections = true;

    // Model paths
    std::string backbonePath = "../models/nanotrack_backbone_sim.onnx";
    std::string neckheadPath = "../models/nanotrack_head_sim.onnx";
    std::string yoloPath = "../models/yolo11n.onnx";
    std::string videoSource = "../videos/car1.mp4";

    // Helper to check if a class should be tracked
    bool shouldTrackClass(const std::string& className) const {
        // If blocked, never track
        if (!blockedClasses.empty() && blockedClasses.count(className)) {
            return false;
        }
        // If allowlist specified, only track those
        if (!allowedClasses.empty()) {
            return allowedClasses.count(className) > 0;
        }
        return true;
    }
};
