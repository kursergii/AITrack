#pragma once

#include <string>
#include <opencv2/opencv.hpp>

#include "tracker_manager.hpp"
#include "camera_motion.hpp"
#include "detector.hpp"

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
    bool running;
    bool paused;
    bool showTrajectoryMap;
    bool detectionEnabled;
    bool autoTrackDetections;
    int detectionInterval;
    int frameCount;
    int targetFps;
    double actualFps;

    static constexpr int MIN_FPS = 1;
    static constexpr int MAX_FPS = 120;
    static constexpr int MAP_WIDTH = 200;
    static constexpr int MAP_HEIGHT = 150;
};
