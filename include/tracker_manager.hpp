/**
 * @file tracker_manager.hpp
 * @brief Multi-object tracker management using NanoTracker and YOLO detections.
 *
 * TrackerManager handles the lifecycle of multiple tracked objects, matching
 * new detections to existing tracks using IoU (Intersection over Union), and
 * maintaining trajectory history with Kalman filter-based motion prediction.
 */

#pragma once

#include <vector>
#include <deque>
#include <set>
#include <mutex>
#include <atomic>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

#include "kalman_predictor.hpp"
#include "camera_motion.hpp"
#include "object_motion.hpp"
#include "detector.hpp"
#include "config.hpp"

/**
 * @class TrackerManager
 * @brief Manages multiple tracked objects with detection-track association.
 *
 * Pipeline stages:
 * 1. Run visual trackers (NanoTracker) on all active tracks
 * 2. Match detections to tracks using greedy IoU matching
 * 3. Update matched tracks with detection positions
 * 4. Continue unmatched tracks with tracker-only updates
 * 5. Create new tracks for unmatched detections
 * 6. Remove tracks that have been lost too long
 */
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

        // Position history for trajectory map
        std::deque<cv::Point2f> trajectory;

        // Status
        bool active;
        int framesLost;
        int framesTracked;
    };

    TrackerManager();
    ~TrackerManager();

    // Configure from Config struct (recommended)
    void configure(const Config& config);

    // Must be called before use - sets model paths for NanoTracker
    void setModelPaths(const std::string& backbone, const std::string& neckhead);

    // Individual configuration setters
    void setMaxLostFrames(int frames) { maxLostFrames = frames; }
    void setMinIoU(float iou) { minIoU = iou; }
    void setReinitThreshold(float thresh) { reinitThreshold = thresh; }
    void setMaxTrajectoryLength(int len) { maxTrajectoryLength = len; }

    // Class filtering
    void setClassFilter(const std::set<std::string>& allowed, const std::set<std::string>& blocked);
    bool shouldTrackClass(const std::string& className) const;

    // Main update - call each frame
    void update(const cv::Mat& frame, const std::vector<Detector::Detection>& detections,
                const CameraMotion& cameraMotion);

    // Update without detections (tracker-only mode)
    void updateTrackers(const cv::Mat& frame, const CameraMotion& cameraMotion);

    // Access tracked objects (thread-safe)
    // Note: For thread safety, use lock() before accessing and unlock() after
    const std::vector<TrackedObject>& getTrackedObjects() const { return trackedObjects; }
    std::vector<TrackedObject>& getTrackedObjects() { return trackedObjects; }
    std::vector<TrackedObject> getTrackedObjectsCopy() const;  // Thread-safe copy
    int getActiveCount() const;

    // Thread safety helpers (use RAII lock_guard instead when possible)
    void lock() const { trackMutex.lock(); }
    void unlock() const { trackMutex.unlock(); }
    std::unique_lock<std::mutex> acquireLock() const { return std::unique_lock<std::mutex>(trackMutex); }

    // Manual track management
    int addTrack(const cv::Mat& frame, const cv::Rect& bbox, const std::string& className = "object");
    void removeTrack(int id);
    void clear();

private:
    // Update pipeline stages (refactored from monolithic update())
    void runTrackers(const cv::Mat& frame);
    void processMatches(const cv::Mat& frame,
                        const std::vector<Detector::Detection>& detections,
                        const std::vector<std::pair<int, int>>& matches,
                        std::vector<bool>& detectionMatched,
                        std::vector<bool>& trackMatched,
                        const CameraMotion& cameraMotion);
    void updateUnmatchedTracks(const std::vector<bool>& trackMatched,
                               const CameraMotion& cameraMotion);
    void createNewTracks(const cv::Mat& frame,
                         const std::vector<Detector::Detection>& detections,
                         const std::vector<bool>& detectionMatched);

    // Matching
    float computeIoU(const cv::Rect& a, const cv::Rect& b) const;
    std::vector<std::pair<int, int>> matchDetectionsToTracks(
        const std::vector<Detector::Detection>& detections) const;

    // Track lifecycle
    void createTrack(const cv::Mat& frame, const Detector::Detection& detection);
    void updateTrack(TrackedObject& track, const cv::Mat& frame,
                     const cv::Rect& newBbox, const CameraMotion& cameraMotion);
    void handleLostTracks();

    // Trajectory management
    void recordTrajectory(TrackedObject& track, const cv::Point2f& center);

    // Generate random color for visualization
    cv::Scalar generateColor() const;

    // Create NanoTracker with configured model paths
    cv::Ptr<cv::TrackerNano> createTracker() const;

    std::vector<TrackedObject> trackedObjects;
    std::atomic<int> nextId{0};

    // Thread safety
    mutable std::mutex trackMutex;

    // Configuration
    int maxLostFrames;
    float minIoU;
    float reinitThreshold;
    int maxTrajectoryLength;

    // Class filtering
    std::set<std::string> allowedClasses;
    std::set<std::string> blockedClasses;

    // NanoTracker model paths
    std::string backbonePath;
    std::string neckheadPath;

    // ID management
    static constexpr int MAX_TRACK_ID = 100000;  // Reset after this to avoid overflow
    int generateTrackId();

    // Color generation
    static constexpr int COLOR_SEED = 42;
    static constexpr int COLOR_MIN = 64;
    static constexpr int COLOR_MAX = 255;
};
