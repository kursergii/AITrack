#pragma once

#include <string>
#include <opencv2/opencv.hpp>

#include "tracker_manager.hpp"
#include "camera_motion.hpp"

class Application {
public:
    Application();
    ~Application();

    bool init(int argc, char** argv);
    int run();

private:
    void processFrame();
    void render();
    void handleInput(char key);
    bool selectROI();

    // Video
    cv::VideoCapture cap;
    cv::Mat frame;
    std::string videoSource;

    // Models
    std::string backbonePath;
    std::string neckheadPath;

    // Multi-object tracking
    TrackerManager trackerManager;

    // Camera motion
    cv::Mat prevGray;
    cv::Mat currGray;
    CameraMotion cameraMotion;

    // State
    bool running;
    bool paused;
    int targetFps;
    double actualFps;

    static constexpr int MIN_FPS = 1;
    static constexpr int MAX_FPS = 120;
};
