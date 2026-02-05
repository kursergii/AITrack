/**
 * @file application.cpp
 * @brief Implementation of the main Application class.
 *
 * Handles video capture, frame processing, async detection, tracking updates,
 * and UI rendering for the multi-object tracking system.
 */

#include "application.hpp"
#include "visualization.hpp"
#include <iostream>

// ============================================================================
// Constructor / Destructor
// ============================================================================

Application::Application()
    : running(false), paused(false), showTrajectoryMap(true),
      detectionEnabled(false), autoTrackDetections(false),
      detectionInterval(10), frameCount(0), targetFps(30), actualFps(0),
      detectionFrameReady(false), detectionsReady(false),
      detectionThreadRunning(false) {
}

Application::~Application() {
    stopDetectionThread();
    cap.release();
    cv::destroyAllWindows();
}

// ============================================================================
// Async Detection Thread
// ============================================================================

/**
 * @brief Start the async detection thread.
 *
 * Detection runs on a separate thread to avoid blocking the main render loop.
 * Frames are submitted periodically and results are retrieved asynchronously.
 */
void Application::startDetectionThread() {
    if (detectionThreadRunning || !detector.isLoaded()) return;

    detectionThreadRunning = true;
    detectionThread = std::thread(&Application::detectionThreadFunc, this);
    std::cout << "Detection thread started" << std::endl;
}

void Application::stopDetectionThread() {
    if (!detectionThreadRunning) return;

    detectionThreadRunning = false;

    // Wake up detection thread so it can exit
    {
        std::lock_guard<std::mutex> lock(detectionMutex);
        detectionFrameReady = true;
        detectionCv.notify_all();
    }

    if (detectionThread.joinable()) {
        detectionThread.join();
    }

    std::cout << "Detection thread stopped" << std::endl;
}

/**
 * @brief Detection thread main function.
 *
 * Waits for frames to be submitted, runs YOLO detection, and stores results
 * for the main thread to retrieve. Uses condition variables for synchronization.
 */
void Application::detectionThreadFunc() {
    while (detectionThreadRunning) {
        cv::Mat frameToProcess;

        // Wait for frame to process
        {
            std::unique_lock<std::mutex> lock(detectionMutex);
            detectionCv.wait(lock, [this] {
                return detectionFrameReady.load() || !detectionThreadRunning.load();
            });

            if (!detectionThreadRunning) break;

            frameToProcess = detectionFrame.clone();
            detectionFrameReady = false;
        }

        // Run detection (this is the expensive operation)
        if (!frameToProcess.empty()) {
            auto detections = detector.detect(frameToProcess);

            // Store results
            {
                std::lock_guard<std::mutex> lock(detectionMutex);
                pendingDetections = std::move(detections);
                detectionsReady = true;
            }
        }
    }
}

// ============================================================================
// Initialization
// ============================================================================

/**
 * @brief Initialize the application.
 * @param argc Command line argument count
 * @param argv Command line arguments: [video_source] [yolo_model] [backbone] [neckhead]
 * @return true if initialization successful, false otherwise
 */
bool Application::init(int argc, char** argv) {
    // Try to load config file
    // Look for config in common locations
    std::vector<std::string> configPaths = {
        "config/aitracker.yaml",
        "../config/aitracker.yaml",
        "aitracker.yaml"
    };

    for (const auto& path : configPaths) {
        if (config.loadFromFile(path)) {
            break;
        }
    }

    // Apply config settings
    backbonePath = config.backbonePath;
    neckheadPath = config.neckheadPath;
    yoloPath = config.yoloPath;
    targetFps = config.targetFps;
    showTrajectoryMap = config.showTrajectoryMap;
    autoTrackDetections = config.autoTrackDetections;
    detectionInterval = config.detectionInterval;

    // Parse command line arguments (override config)
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <video_source> [yolo_model] [backbone] [neckhead]" << std::endl;
        std::cerr << "  video_source: path to video file or '0' for webcam" << std::endl;
        return false;
    }
    videoSource = argv[1];
    if (argc > 2) {
        yoloPath = argv[2];
    }
    if (argc > 4) {
        backbonePath = argv[3];
        neckheadPath = argv[4];
    }

    // Print loaded config
    config.print();

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

    // Configure tracker manager with config settings
    trackerManager.setModelPaths(backbonePath, neckheadPath);
    trackerManager.setMaxLostFrames(config.maxLostFrames);
    trackerManager.setMinIoU(config.minIoU);
    trackerManager.setMaxTrajectoryLength(config.maxTrajectoryLength);
    trackerManager.setClassFilter(config.allowedClasses, config.blockedClasses);

    // Try to load YOLO detector (optional)
    if (detector.load(yoloPath, config.classesPath)) {
        detector.setConfidenceThreshold(config.detectionConfidence);
        detector.setNmsThreshold(config.nmsThreshold);
        detector.setInputSize(config.inputSize);
        detector.tryEnableGPU();  // Try GPU acceleration
        detectionEnabled = true;
        std::cout << "YOLO detection enabled (" << detector.getBackendName() << ")" << std::endl;
    } else {
        std::cout << "YOLO model not found, detection disabled" << std::endl;
        std::cout << "To enable: place model in models/onnx/ folder" << std::endl;
    }

    // Read first frame
    cap >> frame;
    if (frame.empty()) {
        std::cerr << "Error: Cannot read first frame" << std::endl;
        return false;
    }

    // If detection enabled, show detections first, otherwise select ROI manually
    if (detectionEnabled) {
        std::cout << "Press 'd' to toggle detection, 't' to auto-track detections" << std::endl;
    }

    // Select initial ROI (optional with detection)
    if (!detectionEnabled) {
        if (!selectROI()) {
            return false;
        }
    }

    std::cout << "Tracking started." << std::endl;
    std::cout << "Controls: 'q' quit, 'a' add, 'c' clear, 'd' detect, 't' auto-track, 'm' map, SPACE pause" << std::endl;

    running = true;

    // Start async detection thread if detector is loaded
    if (detector.isLoaded()) {
        startDetectionThread();
    }

    return true;
}

/**
 * @brief Allow user to manually select an object to track.
 * @return true if selection was made, false if cancelled
 *
 * Opens an interactive ROI selection dialog. Existing tracks are drawn
 * on the selection frame for reference.
 */
bool Application::selectROI() {
    std::cout << "Select object to track and press ENTER or SPACE" << std::endl;
    std::cout << "Press C to cancel selection" << std::endl;

    // Read a fresh frame for selection
    cv::Mat newFrame;
    cap >> newFrame;
    if (newFrame.empty()) {
        newFrame = frame.clone();
    }

    // Update trackers on the new frame
    cv::Mat newGray;
    cv::cvtColor(newFrame, newGray, cv::COLOR_BGR2GRAY);
    if (!prevGray.empty()) {
        cameraMotion = computeCameraMotion(prevGray, newGray);
    }
    trackerManager.updateTrackers(newFrame, cameraMotion);
    newGray.copyTo(prevGray);

    // Create display frame with existing tracks drawn
    cv::Mat displayFrame = newFrame.clone();
    for (const auto& track : trackerManager.getTrackedObjects()) {
        if (!track.active) continue;
        cv::rectangle(displayFrame, track.bbox, track.color, 2);
        std::string label = "#" + std::to_string(track.id);
        cv::putText(displayFrame, label, cv::Point(track.bbox.x, track.bbox.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, track.color, 2);
    }

    // Use same window as main display
    cv::Rect roi = cv::selectROI("AITrack - Multi-Object Tracker", displayFrame, false, false);

    if (roi.width == 0 || roi.height == 0) {
        std::cout << "Selection cancelled" << std::endl;
        frame = newFrame;  // Keep the frame in sync
        return false;
    }

    // Use the clean frame for adding new track
    frame = newFrame;
    int id = trackerManager.addTrack(frame, roi);
    std::cout << "Added track ID: " << id << std::endl;

    return true;
}

// ============================================================================
// Main Loop
// ============================================================================

/**
 * @brief Main application loop.
 * @return Exit code (0 for success)
 *
 * Runs the capture-process-render loop until user quits or video ends.
 * Uses waitKey delay to maintain target FPS.
 */
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

    stopDetectionThread();
    return 0;
}

/**
 * @brief Process a single frame.
 *
 * 1. Compute camera motion using optical flow
 * 2. Check for async detection results
 * 3. Submit new frame to detection thread periodically
 * 4. Update trackers with or without detections
 */
void Application::processFrame() {
    double timer = cv::getTickCount();
    frameCount++;

    // Compute camera motion
    cv::cvtColor(frame, currGray, cv::COLOR_BGR2GRAY);
    if (!prevGray.empty()) {
        cameraMotion = computeCameraMotion(prevGray, currGray);
    }
    currGray.copyTo(prevGray);

    // Check for async detection results first
    bool hasNewDetections = false;
    {
        std::lock_guard<std::mutex> lock(detectionMutex);
        if (detectionsReady) {
            lastDetections = std::move(pendingDetections);
            detectionsReady = false;
            hasNewDetections = true;
        }
    }

    // Send frame to detection thread periodically (if not already processing)
    if (detectionEnabled && detector.isLoaded() && (frameCount % detectionInterval == 0)) {
        std::lock_guard<std::mutex> lock(detectionMutex);
        if (!detectionFrameReady) {
            detectionFrame = frame.clone();
            detectionFrameReady = true;
            detectionCv.notify_one();
        }
    }

    // Update trackers with or without detections
    if (hasNewDetections && autoTrackDetections && !lastDetections.empty()) {
        trackerManager.update(frame, lastDetections, cameraMotion);
    } else {
        trackerManager.updateTrackers(frame, cameraMotion);
    }

    actualFps = cv::getTickFrequency() / (cv::getTickCount() - timer);
}

// ============================================================================
// Rendering
// ============================================================================

/**
 * @brief Render the current frame with all visualizations.
 *
 * Draws tracked objects with bounding boxes, labels, and predicted paths.
 * Also shows FPS, track count, motion indicators, and trajectory map.
 */
void Application::render() {
    // Don't render if frame is empty
    if (frame.empty()) return;

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
    if (detectionEnabled) {
        trackText += " | Det: " + std::to_string(lastDetections.size());
        if (autoTrackDetections) trackText += " [AUTO]";
    }
    cv::putText(frame, trackText, cv::Point(20, 55),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 100, 0), 1);

    // Draw detections (if not auto-tracking, show as dashed boxes)
    if (detectionEnabled && !autoTrackDetections) {
        renderDetections(lastDetections);
    }

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

    // Draw trajectory map
    renderTrajectoryMap();

    cv::imshow("AITrack - Multi-Object Tracker", frame);
}

// ============================================================================
// Input Handling
// ============================================================================

/**
 * @brief Handle keyboard input.
 * @param key Pressed key character
 *
 * Controls:
 * - q/ESC: Quit
 * - a/r: Add new track (ROI selection)
 * - c: Clear all tracks
 * - +/-: Adjust target FPS
 * - SPACE: Pause/resume
 * - m: Toggle trajectory map
 * - d: Toggle detection
 * - t: Toggle auto-tracking
 * - [/]: Adjust detection threshold
 */
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

        case 'm':
            showTrajectoryMap = !showTrajectoryMap;
            std::cout << "Trajectory map: " << (showTrajectoryMap ? "ON" : "OFF") << std::endl;
            break;

        case 'd':
            if (detector.isLoaded()) {
                detectionEnabled = !detectionEnabled;
                std::cout << "Detection: " << (detectionEnabled ? "ON" : "OFF") << std::endl;
            } else {
                std::cout << "YOLO model not loaded" << std::endl;
            }
            break;

        case 't':
            if (detector.isLoaded()) {
                autoTrackDetections = !autoTrackDetections;
                std::cout << "Auto-track detections: " << (autoTrackDetections ? "ON" : "OFF") << std::endl;
            }
            break;

        case '[':
            detector.setConfidenceThreshold(
                std::max(MIN_CONFIDENCE_THRESHOLD,
                         detector.getConfidenceThreshold() - CONFIDENCE_STEP));
            std::cout << "Detection threshold: " << detector.getConfidenceThreshold() << std::endl;
            break;

        case ']':
            detector.setConfidenceThreshold(
                std::min(MAX_CONFIDENCE_THRESHOLD,
                         detector.getConfidenceThreshold() + CONFIDENCE_STEP));
            std::cout << "Detection threshold: " << detector.getConfidenceThreshold() << std::endl;
            break;
    }
}

/**
 * @brief Render the trajectory map overlay.
 *
 * Shows a minimap in the bottom-left corner with scaled trajectories
 * of all tracked objects. Older positions fade out, current positions
 * are shown as dots with track IDs.
 */
void Application::renderTrajectoryMap() {
    if (!showTrajectoryMap) return;

    const auto& trackedObjects = trackerManager.getTrackedObjects();
    if (trackedObjects.empty()) return;

    // Create semi-transparent map background
    cv::Mat mapBg(MAP_HEIGHT, MAP_WIDTH, CV_8UC3, cv::Scalar(20, 20, 20));

    // Calculate scale to fit trajectories in the map
    float scaleX = static_cast<float>(MAP_WIDTH) / frame.cols;
    float scaleY = static_cast<float>(MAP_HEIGHT) / frame.rows;

    // Draw trajectories for each tracked object
    for (const auto& track : trackedObjects) {
        if (!track.active || track.trajectory.size() < 2) continue;

        // Draw trajectory line
        for (size_t i = 1; i < track.trajectory.size(); i++) {
            cv::Point p1(static_cast<int>(track.trajectory[i-1].x * scaleX),
                        static_cast<int>(track.trajectory[i-1].y * scaleY));
            cv::Point p2(static_cast<int>(track.trajectory[i].x * scaleX),
                        static_cast<int>(track.trajectory[i].y * scaleY));

            // Fade older parts of trajectory
            float alpha = static_cast<float>(i) / track.trajectory.size();
            cv::Scalar color(
                track.color[0] * alpha,
                track.color[1] * alpha,
                track.color[2] * alpha
            );
            cv::line(mapBg, p1, p2, color, 1, cv::LINE_AA);
        }

        // Draw current position as a circle
        if (!track.trajectory.empty()) {
            cv::Point current(
                static_cast<int>(track.trajectory.back().x * scaleX),
                static_cast<int>(track.trajectory.back().y * scaleY));
            cv::circle(mapBg, current, 4, track.color, -1);

            // Draw track ID
            cv::putText(mapBg, std::to_string(track.id),
                       cv::Point(current.x + 5, current.y - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.3, track.color, 1);
        }
    }

    // Draw border
    cv::rectangle(mapBg, cv::Point(0, 0), cv::Point(MAP_WIDTH-1, MAP_HEIGHT-1),
                 cv::Scalar(100, 100, 100), 1);

    // Label
    cv::putText(mapBg, "TRAJECTORY MAP", cv::Point(5, 12),
               cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(150, 150, 150), 1);

    // Overlay map on frame (bottom-left corner)
    int mapX = MAP_OFFSET;
    int mapY = frame.rows - MAP_HEIGHT - MAP_OFFSET;

    // Blend with transparency
    cv::Mat roi = frame(cv::Rect(mapX, mapY, MAP_WIDTH, MAP_HEIGHT));
    cv::addWeighted(mapBg, MAP_ALPHA, roi, 1.0 - MAP_ALPHA, 0, roi);
}

/**
 * @brief Render detection boxes (when auto-tracking is disabled).
 * @param detections Vector of detections to draw
 *
 * Draws corner brackets (instead of full boxes) to distinguish
 * detections from tracked objects. Shows class name and confidence.
 */
void Application::renderDetections(const std::vector<Detector::Detection>& detections) {
    for (const auto& det : detections) {
        // Draw dashed rectangle for detections (not tracked yet)
        cv::Scalar color(0, 200, 255);  // Orange for detections

        // Draw corners instead of full box to distinguish from tracks
        int cornerLen = std::min(20, std::min(det.bbox.width, det.bbox.height) / 3);
        int x1 = det.bbox.x, y1 = det.bbox.y;
        int x2 = det.bbox.x + det.bbox.width, y2 = det.bbox.y + det.bbox.height;

        // Top-left corner
        cv::line(frame, cv::Point(x1, y1), cv::Point(x1 + cornerLen, y1), color, 2);
        cv::line(frame, cv::Point(x1, y1), cv::Point(x1, y1 + cornerLen), color, 2);
        // Top-right corner
        cv::line(frame, cv::Point(x2, y1), cv::Point(x2 - cornerLen, y1), color, 2);
        cv::line(frame, cv::Point(x2, y1), cv::Point(x2, y1 + cornerLen), color, 2);
        // Bottom-left corner
        cv::line(frame, cv::Point(x1, y2), cv::Point(x1 + cornerLen, y2), color, 2);
        cv::line(frame, cv::Point(x1, y2), cv::Point(x1, y2 - cornerLen), color, 2);
        // Bottom-right corner
        cv::line(frame, cv::Point(x2, y2), cv::Point(x2 - cornerLen, y2), color, 2);
        cv::line(frame, cv::Point(x2, y2), cv::Point(x2, y2 - cornerLen), color, 2);

        // Label
        std::string label = det.className + " " +
                           std::to_string(static_cast<int>(det.confidence * 100)) + "%";
        cv::putText(frame, label, cv::Point(det.bbox.x, det.bbox.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
    }
}
