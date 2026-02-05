/**
 * @file config.hpp
 * @brief Centralized configuration for all tunable parameters.
 *
 * Provides default values and runtime configuration for tracker settings,
 * detection thresholds, class filtering, display options, and model paths.
 * Can load settings from YAML config file.
 */

#pragma once

#include <string>
#include <vector>
#include <set>
#include <opencv2/core.hpp>
#include <iostream>

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
    std::string classesPath;  // Path to class names file (one per line); empty = COCO defaults
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

    // Load configuration from YAML file (OpenCV YAML requires %YAML:1.0 header)
    bool loadFromFile(const std::string& configPath) {
        cv::FileStorage fs;
        try {
            fs.open(configPath, cv::FileStorage::READ);
        } catch (const cv::Exception&) {
            std::cerr << "Warning: Invalid config file: " << configPath << std::endl;
            return false;
        }
        if (!fs.isOpened()) {
            std::cerr << "Warning: Could not open config file: " << configPath << std::endl;
            return false;
        }

        std::cout << "Loading config from: " << configPath << std::endl;

        // Tracker settings
        if (!fs["tracker"].empty()) {
            cv::FileNode tracker = fs["tracker"];
            if (!tracker["max_lost_frames"].empty()) tracker["max_lost_frames"] >> maxLostFrames;
            if (!tracker["min_iou"].empty()) tracker["min_iou"] >> minIoU;
            if (!tracker["reinit_threshold"].empty()) tracker["reinit_threshold"] >> reinitThreshold;
            if (!tracker["max_trajectory_length"].empty()) tracker["max_trajectory_length"] >> maxTrajectoryLength;
        }

        // Detection settings
        if (!fs["detection"].empty()) {
            cv::FileNode detection = fs["detection"];
            if (!detection["confidence"].empty()) detection["confidence"] >> detectionConfidence;
            if (!detection["nms_threshold"].empty()) detection["nms_threshold"] >> nmsThreshold;
            if (!detection["interval"].empty()) detection["interval"] >> detectionInterval;
            if (!detection["input_size"].empty()) detection["input_size"] >> inputSize;
        }

        // Display settings
        if (!fs["display"].empty()) {
            cv::FileNode display = fs["display"];
            if (!display["target_fps"].empty()) display["target_fps"] >> targetFps;
            if (!display["show_trajectory_map"].empty()) display["show_trajectory_map"] >> showTrajectoryMap;
            if (!display["auto_track_detections"].empty()) display["auto_track_detections"] >> autoTrackDetections;
        }

        // Model paths
        if (!fs["models"].empty()) {
            cv::FileNode models = fs["models"];
            if (!models["yolo"].empty()) models["yolo"] >> yoloPath;
            if (!models["nanotrack_backbone"].empty()) models["nanotrack_backbone"] >> backbonePath;
            if (!models["nanotrack_head"].empty()) models["nanotrack_head"] >> neckheadPath;
            if (!models["classes_file"].empty()) models["classes_file"] >> classesPath;
        }

        // Class filtering
        if (!fs["classes"].empty()) {
            cv::FileNode classes = fs["classes"];

            if (!classes["allowed"].empty()) {
                allowedClasses.clear();
                cv::FileNode allowed = classes["allowed"];
                for (auto it = allowed.begin(); it != allowed.end(); ++it) {
                    std::string cls;
                    *it >> cls;
                    if (!cls.empty()) allowedClasses.insert(cls);
                }
            }

            if (!classes["blocked"].empty()) {
                blockedClasses.clear();
                cv::FileNode blocked = classes["blocked"];
                for (auto it = blocked.begin(); it != blocked.end(); ++it) {
                    std::string cls;
                    *it >> cls;
                    if (!cls.empty()) blockedClasses.insert(cls);
                }
            }
        }

        fs.release();
        return true;
    }

    // Print current configuration
    void print() const {
        std::cout << "=== AITracker Configuration ===" << std::endl;
        std::cout << "Tracker:" << std::endl;
        std::cout << "  max_lost_frames: " << maxLostFrames << std::endl;
        std::cout << "  min_iou: " << minIoU << std::endl;
        std::cout << "  max_trajectory_length: " << maxTrajectoryLength << std::endl;
        std::cout << "Detection:" << std::endl;
        std::cout << "  confidence: " << detectionConfidence << std::endl;
        std::cout << "  interval: " << detectionInterval << std::endl;
        std::cout << "Display:" << std::endl;
        std::cout << "  target_fps: " << targetFps << std::endl;
        std::cout << "  auto_track: " << (autoTrackDetections ? "yes" : "no") << std::endl;
        std::cout << "Models:" << std::endl;
        std::cout << "  yolo: " << yoloPath << std::endl;
        std::cout << "  backbone: " << backbonePath << std::endl;
        std::cout << "  head: " << neckheadPath << std::endl;
        if (!allowedClasses.empty()) {
            std::cout << "Allowed classes: ";
            for (const auto& c : allowedClasses) std::cout << c << " ";
            std::cout << std::endl;
        }
        std::cout << "===============================" << std::endl;
    }
};
