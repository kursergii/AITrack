/**
 * @file application.hpp
 * @brief Main application class for AITrack multi-object tracking system.
 *
 * This application combines YOLO object detection with NanoTracker visual tracking
 * to provide real-time multi-object tracking with motion prediction and trajectory
 * visualization. Detection runs asynchronously on a separate thread to maintain
 * smooth video playback.
 */

#pragma once

#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <queue>
#include <opencv2/opencv.hpp>

#include "tracker_manager.hpp"
#include "camera_motion.hpp"
#include "detector.hpp"

/**
 * @class Application
 * @brief Main application controller for video object tracking.
 *
 * Manages the main loop, video capture, detection/tracking pipeline, and UI rendering.
 * Uses async detection to avoid blocking the main render loop during YOLO inference.
 */
class Application {
public:
    Application();
    ~Application();

    bool init(int argc, char** argv);
    int run();

private:
    void processFrame();
    void render();
    void renderTrajectoryMap();
    void renderDetections(const std::vector<Detector::Detection>& detections);
    void handleInput(char key);
    bool selectROI();

    // Threading (async detection only)
    void detectionThreadFunc();
    void startDetectionThread();
    void stopDetectionThread();

    // Video
    cv::VideoCapture cap;
    cv::Mat frame;
    std::string videoSource;

    // Models
    std::string backbonePath;
    std::string neckheadPath;
    std::string yoloPath;

    // Multi-object tracking
    TrackerManager trackerManager;

    // Object detection
    Detector detector;
    std::vector<Detector::Detection> lastDetections;

    // Camera motion
    cv::Mat prevGray;
    cv::Mat currGray;
    CameraMotion cameraMotion;

    // State
    std::atomic<bool> running;
    bool paused;
    bool showTrajectoryMap;
    std::atomic<bool> detectionEnabled;
    std::atomic<bool> autoTrackDetections;
    int detectionInterval;
    std::atomic<int> frameCount;
    int targetFps;
    double actualFps;

    // Threading - Async Detection
    std::thread detectionThread;
    std::mutex detectionMutex;
    std::condition_variable detectionCv;
    cv::Mat detectionFrame;              // Frame to process
    std::atomic<bool> detectionFrameReady;
    std::vector<Detector::Detection> pendingDetections;
    std::atomic<bool> detectionsReady;
    std::atomic<bool> detectionThreadRunning;

    static constexpr int MIN_FPS = 1;
    static constexpr int MAX_FPS = 120;
    static constexpr int MAP_WIDTH = 200;
    static constexpr int MAP_HEIGHT = 150;

    // Detection threshold limits
    static constexpr float MIN_CONFIDENCE_THRESHOLD = 0.1f;
    static constexpr float MAX_CONFIDENCE_THRESHOLD = 0.9f;
    static constexpr float CONFIDENCE_STEP = 0.1f;

    // Trajectory map
    static constexpr int MAP_OFFSET = 10;
    static constexpr double MAP_ALPHA = 0.8;
};
