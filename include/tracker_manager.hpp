#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

#include "kalman_predictor.hpp"
#include "camera_motion.hpp"
#include "object_motion.hpp"
#include "detector.hpp"

class TrackerManager {
public:
    struct TrackedObject {
        int id;
        cv::Rect bbox;
        cv::Rect prevBbox;
        std::string className;
        float confidence;
        cv::Scalar color;

        // Tracking state
        cv::Ptr<cv::TrackerNano> tracker;
        KalmanPredictor predictor;
        ObjectMotion motion;

        // Status
        bool active;
        int framesLost;
        int framesTracked;
    };

    TrackerManager();
    ~TrackerManager();

    // Must be called before use - sets model paths for NanoTracker
    void setModelPaths(const std::string& backbone, const std::string& neckhead);

    // Configuration
    void setMaxLostFrames(int frames) { maxLostFrames = frames; }
    void setMinIoU(float iou) { minIoU = iou; }

    // Main update - call each frame
    void update(const cv::Mat& frame, const std::vector<Detector::Detection>& detections,
                const CameraMotion& cameraMotion);

    // Update without detections (tracker-only mode)
    void updateTrackers(const cv::Mat& frame, const CameraMotion& cameraMotion);

    // Access tracked objects
    const std::vector<TrackedObject>& getTrackedObjects() const { return trackedObjects; }
    std::vector<TrackedObject>& getTrackedObjects() { return trackedObjects; }
    int getActiveCount() const;

    // Manual track management
    int addTrack(const cv::Mat& frame, const cv::Rect& bbox, const std::string& className = "object");
    void removeTrack(int id);
    void clear();

private:
    // Matching
    float computeIoU(const cv::Rect& a, const cv::Rect& b) const;
    std::vector<std::pair<int, int>> matchDetectionsToTracks(
        const std::vector<Detector::Detection>& detections) const;

    // Track lifecycle
    void createTrack(const cv::Mat& frame, const Detector::Detection& detection);
    void updateTrack(TrackedObject& track, const cv::Mat& frame,
                     const cv::Rect& newBbox, const CameraMotion& cameraMotion);
    void handleLostTracks();

    // Generate random color for visualization
    cv::Scalar generateColor() const;

    // Create NanoTracker with configured model paths
    cv::Ptr<cv::TrackerNano> createTracker() const;

    std::vector<TrackedObject> trackedObjects;
    int nextId;

    // Configuration
    int maxLostFrames;
    float minIoU;

    // NanoTracker model paths
    std::string backbonePath;
    std::string neckheadPath;
};
