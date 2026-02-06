/**
 * @file tracker_manager.cpp
 * @brief Implementation of multi-object tracker management.
 *
 * Handles the lifecycle of tracked objects including creation, updates,
 * detection-track matching, and removal of lost tracks.
 */

#include "tracker_manager.hpp"
#include <algorithm>
#include <random>
#include <iostream>

// ============================================================================
// Constructor / Destructor
// ============================================================================

TrackerManager::TrackerManager()
    : maxLostFrames(30), minIoU(0.3f), reinitThreshold(0.7f),
      maxTrajectoryLength(300) {
    nextId.store(0);
}

TrackerManager::~TrackerManager() {
}

// ============================================================================
// Configuration
// ============================================================================

/// Generate unique track ID with wraparound to prevent overflow
int TrackerManager::generateTrackId() {
    int id = nextId.fetch_add(1);
    // Reset to avoid overflow (IDs may repeat after MAX_TRACK_ID tracks)
    if (id >= MAX_TRACK_ID) {
        nextId.store(0);
    }
    return id;
}

std::vector<TrackerManager::TrackedObject> TrackerManager::getTrackedObjectsCopy() const {
    std::lock_guard<std::mutex> lock(trackMutex);
    return trackedObjects;
}

void TrackerManager::configure(const Config& config) {
    maxLostFrames = config.maxLostFrames;
    minIoU = config.minIoU;
    reinitThreshold = config.reinitThreshold;
    maxTrajectoryLength = config.maxTrajectoryLength;
    allowedClasses = config.allowedClasses;
    blockedClasses = config.blockedClasses;
    backbonePath = config.backbonePath;
    neckheadPath = config.neckheadPath;
}

void TrackerManager::setModelPaths(const std::string& backbone, const std::string& neckhead) {
    backbonePath = backbone;
    neckheadPath = neckhead;
}

void TrackerManager::setClassFilter(const std::set<std::string>& allowed,
                                     const std::set<std::string>& blocked) {
    allowedClasses = allowed;
    blockedClasses = blocked;
}

bool TrackerManager::shouldTrackClass(const std::string& className) const {
    if (!blockedClasses.empty() && blockedClasses.count(className)) {
        return false;
    }
    if (!allowedClasses.empty()) {
        return allowedClasses.count(className) > 0;
    }
    return true;
}

// ============================================================================
// Tracker Factory
// ============================================================================

/// Create a new NanoTracker instance with configured model paths
cv::Ptr<cv::TrackerNano> TrackerManager::createTracker() const {
    if (backbonePath.empty() || neckheadPath.empty()) {
        std::cerr << "TrackerManager: Model paths not set. Call setModelPaths() first." << std::endl;
        return nullptr;
    }

    cv::TrackerNano::Params params;
    params.backbone = backbonePath;
    params.neckhead = neckheadPath;

    try {
        return cv::TrackerNano::create(params);
    } catch (const cv::Exception& e) {
        std::cerr << "Error creating NanoTracker: " << e.what() << std::endl;
        return nullptr;
    }
}

// ============================================================================
// Update Pipeline
// ============================================================================

/// Record position in trajectory history with max length limit
void TrackerManager::recordTrajectory(TrackedObject& track, const cv::Point2f& center) {
    track.trajectory.push_back(center);
    if (static_cast<int>(track.trajectory.size()) > maxTrajectoryLength) {
        track.trajectory.pop_front();
    }
}

/// Stage 1: Run visual trackers on all active tracks
void TrackerManager::runTrackers(const cv::Mat& frame) {
    for (auto& track : trackedObjects) {
        if (!track.active || !track.tracker) continue;

        cv::Rect newBbox;
        bool success = track.tracker->update(frame, newBbox);

        if (success) {
            track.prevBbox = track.bbox;
            track.bbox = newBbox;
        }
    }
}

/// Stage 3: Update tracks that matched detections
void TrackerManager::processMatches(const cv::Mat& frame,
                                     const std::vector<Detector::Detection>& detections,
                                     const std::vector<std::pair<int, int>>& matches,
                                     std::vector<bool>& detectionMatched,
                                     std::vector<bool>& trackMatched,
                                     const CameraMotion& cameraMotion) {
    for (const auto& match : matches) {
        int detIdx = match.first;
        int trackIdx = match.second;

        // Boundary checks
        if (detIdx < 0 || detIdx >= static_cast<int>(detections.size())) continue;
        if (trackIdx < 0 || trackIdx >= static_cast<int>(trackedObjects.size())) continue;

        detectionMatched[detIdx] = true;
        trackMatched[trackIdx] = true;

        updateTrack(trackedObjects[trackIdx], frame,
                    detections[detIdx].bbox, cameraMotion);
        trackedObjects[trackIdx].confidence = detections[detIdx].confidence;
        trackedObjects[trackIdx].framesLost = 0;
    }
}

/// Stage 4: Continue unmatched tracks using tracker-only updates
void TrackerManager::updateUnmatchedTracks(const std::vector<bool>& trackMatched,
                                            const CameraMotion& cameraMotion) {
    for (size_t i = 0; i < trackedObjects.size(); i++) {
        if (!trackMatched[i] && trackedObjects[i].active) {
            trackedObjects[i].framesLost++;
            trackedObjects[i].framesTracked++;

            trackedObjects[i].motion = computeObjectMotion(
                trackedObjects[i].prevBbox,
                trackedObjects[i].bbox,
                cameraMotion);

            cv::Point2f center(
                trackedObjects[i].bbox.x + trackedObjects[i].bbox.width / 2.0f,
                trackedObjects[i].bbox.y + trackedObjects[i].bbox.height / 2.0f);
            trackedObjects[i].predictor.update(center, trackedObjects[i].motion.envMotion);
            recordTrajectory(trackedObjects[i], center);
        }
    }
}

/// Stage 5: Create new tracks for unmatched detections
/// Skips detections that are near an existing track (even if IoU didn't match)
void TrackerManager::createNewTracks(const cv::Mat& frame,
                                      const std::vector<Detector::Detection>& detections,
                                      const std::vector<bool>& detectionMatched) {
    for (size_t i = 0; i < detections.size(); i++) {
        if (detectionMatched[i]) continue;
        if (!shouldTrackClass(detections[i].className)) continue;

        // Check if any existing track is already near this detection
        cv::Point2f detCenter(detections[i].bbox.x + detections[i].bbox.width / 2.0f,
                              detections[i].bbox.y + detections[i].bbox.height / 2.0f);
        bool tooClose = false;
        for (const auto& track : trackedObjects) {
            if (!track.active) continue;
            cv::Point2f trackCenter(track.bbox.x + track.bbox.width / 2.0f,
                                    track.bbox.y + track.bbox.height / 2.0f);
            float dist = cv::norm(detCenter - trackCenter);
            float maxDim = static_cast<float>(std::max({
                detections[i].bbox.width, detections[i].bbox.height,
                track.bbox.width, track.bbox.height
            }));
            if (dist < maxDim * 2.0f) {
                tooClose = true;
                break;
            }
        }

        if (!tooClose) {
            createTrack(frame, detections[i]);
        }
    }
}

/**
 * @brief Main update with detections (full pipeline).
 *
 * Runs all 6 stages:
 * 1. Run visual trackers
 * 2. Match detections to tracks (IoU-based greedy matching)
 * 3. Update matched tracks with detection positions
 * 4. Continue unmatched tracks with tracker positions
 * 5. Create new tracks for unmatched detections
 * 6. Remove lost tracks
 */
void TrackerManager::update(const cv::Mat& frame,
                            const std::vector<Detector::Detection>& detections,
                            const CameraMotion& cameraMotion) {
    std::lock_guard<std::mutex> lock(trackMutex);

    if (frame.empty()) return;

    // Stage 1: Update all existing trackers
    runTrackers(frame);

    // Stage 2: Match detections to tracks
    auto matches = matchDetectionsToTracks(detections);
    std::vector<bool> detectionMatched(detections.size(), false);
    std::vector<bool> trackMatched(trackedObjects.size(), false);

    // Stage 3: Process matched pairs
    processMatches(frame, detections, matches, detectionMatched, trackMatched, cameraMotion);

    // Stage 4: Update unmatched tracks (tracker-only)
    updateUnmatchedTracks(trackMatched, cameraMotion);

    // Stage 5: Create tracks for unmatched detections
    createNewTracks(frame, detections, detectionMatched);

    // Stage 6: Clean up lost tracks
    handleLostTracks();
}

/// Update without detections (tracker-only mode)
void TrackerManager::updateTrackers(const cv::Mat& frame, const CameraMotion& cameraMotion) {
    std::lock_guard<std::mutex> lock(trackMutex);

    if (frame.empty()) return;

    for (auto& track : trackedObjects) {
        if (!track.active || !track.tracker) continue;

        cv::Rect newBbox;
        bool success = track.tracker->update(frame, newBbox);

        if (success) {
            track.prevBbox = track.bbox;
            track.bbox = newBbox;
            track.framesTracked++;

            // Update motion
            track.motion = computeObjectMotion(track.prevBbox, track.bbox, cameraMotion);

            cv::Point2f center(
                track.bbox.x + track.bbox.width / 2.0f,
                track.bbox.y + track.bbox.height / 2.0f);
            track.predictor.update(center, track.motion.envMotion);

            recordTrajectory(track, center);
        } else {
            track.framesLost++;
        }
    }

    handleLostTracks();
}

// ============================================================================
// Track Management
// ============================================================================

/// Count currently active tracks (thread-safe)
int TrackerManager::getActiveCount() const {
    std::lock_guard<std::mutex> lock(trackMutex);
    int count = 0;
    for (const auto& track : trackedObjects) {
        if (track.active) count++;
    }
    return count;
}

/// Manually add a new track from ROI selection
int TrackerManager::addTrack(const cv::Mat& frame, const cv::Rect& bbox,
                              const std::string& className) {
    std::lock_guard<std::mutex> lock(trackMutex);

    // Validate input
    if (frame.empty()) return -1;
    if (bbox.width <= 0 || bbox.height <= 0) return -1;
    if (bbox.x < 0 || bbox.y < 0) return -1;
    if (bbox.x + bbox.width > frame.cols || bbox.y + bbox.height > frame.rows) return -1;

    TrackedObject track;
    track.id = generateTrackId();
    track.bbox = bbox;
    track.prevBbox = bbox;
    track.className = className;
    track.confidence = 1.0f;
    track.color = generateColor();
    track.active = true;
    track.framesLost = 0;
    track.framesTracked = 0;

    // Create NanoTracker
    track.tracker = createTracker();
    if (track.tracker) {
        track.tracker->init(frame, bbox);
    }

    // Initialize predictor and trajectory
    cv::Point2f center(bbox.x + bbox.width / 2.0f, bbox.y + bbox.height / 2.0f);
    track.predictor.init(center);
    track.trajectory.push_back(center);

    int id = track.id;
    trackedObjects.push_back(std::move(track));
    return id;
}

void TrackerManager::removeTrack(int id) {
    std::lock_guard<std::mutex> lock(trackMutex);
    auto it = std::find_if(trackedObjects.begin(), trackedObjects.end(),
                           [id](const TrackedObject& t) { return t.id == id; });
    if (it != trackedObjects.end()) {
        trackedObjects.erase(it);
    }
}

void TrackerManager::clear() {
    std::lock_guard<std::mutex> lock(trackMutex);
    trackedObjects.clear();
    nextId.store(0);
}

// ============================================================================
// Detection-Track Matching
// ============================================================================

/// Compute Intersection over Union between two bounding boxes
float TrackerManager::computeIoU(const cv::Rect& a, const cv::Rect& b) const {
    int x1 = std::max(a.x, b.x);
    int y1 = std::max(a.y, b.y);
    int x2 = std::min(a.x + a.width, b.x + b.width);
    int y2 = std::min(a.y + a.height, b.y + b.height);

    if (x2 <= x1 || y2 <= y1) {
        return 0.0f;
    }

    float intersection = static_cast<float>((x2 - x1) * (y2 - y1));
    float areaA = static_cast<float>(a.width * a.height);
    float areaB = static_cast<float>(b.width * b.height);
    float unionArea = areaA + areaB - intersection;

    return intersection / unionArea;
}

/**
 * @brief Match detections to tracks using greedy IoU matching.
 * @return Vector of (detection_idx, track_idx) pairs
 *
 * Builds IoU matrix and greedily assigns detections to tracks,
 * always picking the highest IoU match above the minimum threshold.
 */
std::vector<std::pair<int, int>> TrackerManager::matchDetectionsToTracks(
    const std::vector<Detector::Detection>& detections) const {

    std::vector<std::pair<int, int>> matches;

    if (detections.empty() || trackedObjects.empty()) {
        return matches;
    }

    // Compute IoU matrix
    std::vector<std::vector<float>> iouMatrix(detections.size(),
        std::vector<float>(trackedObjects.size(), 0.0f));

    for (size_t d = 0; d < detections.size(); d++) {
        for (size_t t = 0; t < trackedObjects.size(); t++) {
            if (trackedObjects[t].active) {
                iouMatrix[d][t] = computeIoU(detections[d].bbox, trackedObjects[t].bbox);
            }
        }
    }

    // Greedy matching
    std::vector<bool> detUsed(detections.size(), false);
    std::vector<bool> trackUsed(trackedObjects.size(), false);

    while (true) {
        float bestIoU = minIoU;
        int bestDet = -1;
        int bestTrack = -1;

        for (size_t d = 0; d < detections.size(); d++) {
            if (detUsed[d]) continue;
            for (size_t t = 0; t < trackedObjects.size(); t++) {
                if (trackUsed[t] || !trackedObjects[t].active) continue;
                if (iouMatrix[d][t] > bestIoU) {
                    bestIoU = iouMatrix[d][t];
                    bestDet = static_cast<int>(d);
                    bestTrack = static_cast<int>(t);
                }
            }
        }

        if (bestDet < 0) break;

        matches.emplace_back(bestDet, bestTrack);
        detUsed[bestDet] = true;
        trackUsed[bestTrack] = true;
    }

    return matches;
}

// ============================================================================
// Track Lifecycle
// ============================================================================

/// Create a new track from a detection
void TrackerManager::createTrack(const cv::Mat& frame, const Detector::Detection& detection) {
    // Validate detection bbox
    if (detection.bbox.width <= 0 || detection.bbox.height <= 0) return;
    if (detection.bbox.x < 0 || detection.bbox.y < 0) return;
    if (!frame.empty() && (detection.bbox.x + detection.bbox.width > frame.cols ||
                           detection.bbox.y + detection.bbox.height > frame.rows)) return;

    TrackedObject track;
    track.id = generateTrackId();
    track.bbox = detection.bbox;
    track.prevBbox = detection.bbox;
    track.className = detection.className;
    track.confidence = detection.confidence;
    track.color = generateColor();
    track.active = true;
    track.framesLost = 0;
    track.framesTracked = 0;

    // Create NanoTracker
    track.tracker = createTracker();
    if (track.tracker) {
        track.tracker->init(frame, detection.bbox);
    }

    // Initialize predictor and trajectory
    cv::Point2f center(detection.bbox.x + detection.bbox.width / 2.0f,
                       detection.bbox.y + detection.bbox.height / 2.0f);
    track.predictor.init(center);
    track.trajectory.push_back(center);

    std::cout << "Auto-track: created track #" << track.id << " (" << detection.className << ")" << std::endl;
    trackedObjects.push_back(std::move(track));
}

/**
 * @brief Update an existing track with new detection position.
 *
 * Reinitializes the tracker if it has drifted too far from the detection
 * (IoU below reinitThreshold). Updates motion, predictor, and trajectory.
 */
void TrackerManager::updateTrack(TrackedObject& track, const cv::Mat& frame,
                                  const cv::Rect& newBbox, const CameraMotion& cameraMotion) {
    // Check if tracker has drifted significantly from detection
    float iou = computeIoU(track.bbox, newBbox);
    bool needsReinit = (iou < reinitThreshold) || !track.tracker;

    track.prevBbox = track.bbox;
    track.bbox = newBbox;
    track.framesTracked++;

    // Only reinitialize tracker if drifted beyond threshold (expensive operation)
    if (needsReinit) {
        track.tracker = createTracker();
        if (track.tracker) {
            track.tracker->init(frame, newBbox);
        }
    }

    // Update motion
    track.motion = computeObjectMotion(track.prevBbox, track.bbox, cameraMotion);

    // Update predictor and trajectory
    cv::Point2f center(newBbox.x + newBbox.width / 2.0f,
                       newBbox.y + newBbox.height / 2.0f);
    track.predictor.update(center, track.motion.envMotion);
    recordTrajectory(track, center);
}

/// Mark tracks as inactive if lost too long, then remove inactive tracks
void TrackerManager::handleLostTracks() {
    for (auto& track : trackedObjects) {
        if (track.active && track.framesLost > maxLostFrames) {
            track.active = false;
        }
    }

    // Remove inactive tracks
    trackedObjects.erase(
        std::remove_if(trackedObjects.begin(), trackedObjects.end(),
                       [](const TrackedObject& t) { return !t.active; }),
        trackedObjects.end());
}

/// Generate a random color for track visualization
cv::Scalar TrackerManager::generateColor() const {
    static std::mt19937 rng(COLOR_SEED);
    std::uniform_int_distribution<int> dist(COLOR_MIN, COLOR_MAX);
    return cv::Scalar(dist(rng), dist(rng), dist(rng));
}
