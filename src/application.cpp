#include "application.hpp"
#include "visualization.hpp"
#include <iostream>

Application::Application()
    : running(false), paused(false), targetFps(30), actualFps(0) {
}

Application::~Application() {
    cap.release();
    cv::destroyAllWindows();
}

bool Application::init(int argc, char** argv) {
    // Parse arguments
    videoSource = "../videos/car1.mp4";
    backbonePath = "../models/nanotrack_backbone_sim.onnx";
    neckheadPath = "../models/nanotrack_head_sim.onnx";

    if (argc > 1) {
        videoSource = argv[1];
    }
    if (argc > 3) {
        backbonePath = argv[2];
        neckheadPath = argv[3];
    }

    // Open video
    if (videoSource == "0") {
        cap.open(0);
    } else {
        cap.open(videoSource);
    }

    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open video source: " << videoSource << std::endl;
        return false;
    }

    // Configure tracker manager
    trackerManager.setModelPaths(backbonePath, neckheadPath);

    // Read first frame
    cap >> frame;
    if (frame.empty()) {
        std::cerr << "Error: Cannot read first frame" << std::endl;
        return false;
    }

    // Select initial ROI
    if (!selectROI()) {
        return false;
    }

    std::cout << "Tracking started." << std::endl;
    std::cout << "Controls: 'q' quit, 'r' reselect, 'a' add track, 'c' clear all, '+'/'-' adjust FPS, SPACE pause" << std::endl;

    running = true;
    return true;
}

bool Application::selectROI() {
    std::cout << "Select object to track and press ENTER or SPACE" << std::endl;
    std::cout << "Press C to cancel selection" << std::endl;

    cv::Rect roi = cv::selectROI("AITrack - Select Object", frame, false, false);

    if (roi.width == 0 || roi.height == 0) {
        std::cerr << "No ROI selected" << std::endl;
        return false;
    }

    int id = trackerManager.addTrack(frame, roi);
    std::cout << "Added track ID: " << id << std::endl;
    prevGray.release();

    return true;
}

int Application::run() {
    while (running) {
        if (!paused) {
            cap >> frame;
            if (frame.empty()) {
                std::cout << "End of video" << std::endl;
                break;
            }
            processFrame();
        }

        render();

        int delay = 1000 / targetFps;
        char key = static_cast<char>(cv::waitKey(delay));
        handleInput(key);
    }

    return 0;
}

void Application::processFrame() {
    double timer = cv::getTickCount();

    // Compute camera motion
    cv::cvtColor(frame, currGray, cv::COLOR_BGR2GRAY);
    if (!prevGray.empty()) {
        cameraMotion = computeCameraMotion(prevGray, currGray);
    }
    currGray.copyTo(prevGray);

    // Update all trackers (handles motion and prediction internally)
    trackerManager.updateTrackers(frame, cameraMotion);

    actualFps = cv::getTickFrequency() / (cv::getTickCount() - timer);
}

void Application::render() {
    const auto& trackedObjects = trackerManager.getTrackedObjects();
    int activeCount = trackerManager.getActiveCount();

    // Draw each tracked object
    for (const auto& track : trackedObjects) {
        if (!track.active) continue;

        // Bounding box with track color
        cv::rectangle(frame, track.bbox, track.color, 2);

        // Label with ID and class
        std::string label = "#" + std::to_string(track.id);
        if (!track.className.empty() && track.className != "object") {
            label += " " + track.className;
        }
        cv::putText(frame, label, cv::Point(track.bbox.x, track.bbox.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, track.color, 2);

        // Predicted path
        cv::Point2f currentCenter(track.bbox.x + track.bbox.width / 2.0f,
                                  track.bbox.y + track.bbox.height / 2.0f);
        auto predictedPath = track.predictor.predictPath(KalmanPredictor::PREDICTION_FRAMES);
        float confidence = track.predictor.getConfidence();
        drawPrediction(frame, currentCenter, predictedPath, confidence);
    }

    // FPS and track count
    std::string fpsText = "FPS: " + std::to_string(targetFps) +
                          " (actual: " + std::to_string(static_cast<int>(actualFps)) + ")";
    cv::putText(frame, fpsText, cv::Point(20, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);

    std::string trackText = "Tracks: " + std::to_string(activeCount);
    cv::putText(frame, trackText, cv::Point(20, 55),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 100, 0), 1);

    // Camera motion indicator
    cv::Point camIndicatorPos(frame.cols - 60, 60);
    drawMotionIndicator(frame, cameraMotion.direction, cameraMotion.magnitude,
                        camIndicatorPos, cv::Scalar(0, 255, 255));
    cv::putText(frame, "CAM: " + cameraMotion.directionName,
                cv::Point(frame.cols - 130, 115),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 255), 1);

    // Object motion indicator for first active track
    if (!trackedObjects.empty() && trackedObjects[0].active) {
        const auto& track = trackedObjects[0];
        cv::Point objIndicatorPos(frame.cols - 60, 180);
        drawMotionIndicator(frame, track.motion.envMotion, track.motion.magnitude,
                            objIndicatorPos, cv::Scalar(255, 0, 255), 2.0f);
        cv::putText(frame, "OBJ: " + track.motion.directionName,
                    cv::Point(frame.cols - 130, 235),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 0, 255), 1);
    }

    // Paused indicator
    if (paused) {
        cv::putText(frame, "PAUSED", cv::Point(20, 75),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 165, 255), 2);
    }

    // No tracks indicator
    if (activeCount == 0) {
        cv::putText(frame, "No tracks - press 'a' to add", cv::Point(20, 75),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
    }

    cv::imshow("AITrack - Multi-Object Tracker", frame);
}

void Application::handleInput(char key) {
    switch (key) {
        case 'q':
        case 27:  // ESC
            running = false;
            break;

        case 'a':
        case 'r':
            selectROI();
            break;

        case 'c':
            trackerManager.clear();
            std::cout << "Cleared all tracks" << std::endl;
            break;

        case '+':
        case '=':
            targetFps = std::min(targetFps + 5, MAX_FPS);
            std::cout << "Target FPS: " << targetFps << std::endl;
            break;

        case '-':
        case '_':
            targetFps = std::max(targetFps - 5, MIN_FPS);
            std::cout << "Target FPS: " << targetFps << std::endl;
            break;

        case ' ':
            paused = !paused;
            std::cout << (paused ? "Paused" : "Resumed") << std::endl;
            break;
    }
}
