#include "tracker_manager.hpp"
#include <algorithm>
#include <random>
#include <iostream>

TrackerManager::TrackerManager()
    : nextId(0), maxLostFrames(30), minIoU(0.3f) {
}

TrackerManager::~TrackerManager() {
}

void TrackerManager::setModelPaths(const std::string& backbone, const std::string& neckhead) {
    backbonePath = backbone;
    neckheadPath = neckhead;
}

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

void TrackerManager::update(const cv::Mat& frame,
                            const std::vector<Detector::Detection>& detections,
                            const CameraMotion& cameraMotion) {
    // First, update all existing trackers
    for (auto& track : trackedObjects) {
        if (!track.active || !track.tracker) continue;

        cv::Rect newBbox;
        bool success = track.tracker->update(frame, newBbox);

        if (success) {
            track.prevBbox = track.bbox;
            track.bbox = newBbox;
        }
    }

    // Match detections to existing tracks
    auto matches = matchDetectionsToTracks(detections);
    std::vector<bool> detectionMatched(detections.size(), false);
    std::vector<bool> trackMatched(trackedObjects.size(), false);

    // Process matches
    for (const auto& match : matches) {
        int detIdx = match.first;
        int trackIdx = match.second;

        detectionMatched[detIdx] = true;
        trackMatched[trackIdx] = true;

        // Update track with detection
        updateTrack(trackedObjects[trackIdx], frame,
                    detections[detIdx].bbox, cameraMotion);
        trackedObjects[trackIdx].confidence = detections[detIdx].confidence;
        trackedObjects[trackIdx].framesLost = 0;
    }

    // Update motion for unmatched tracks (tracker only)
    for (size_t i = 0; i < trackedObjects.size(); i++) {
        if (!trackMatched[i] && trackedObjects[i].active) {
            trackedObjects[i].framesLost++;

            // Still update motion based on tracker
            trackedObjects[i].motion = computeObjectMotion(
                trackedObjects[i].prevBbox,
                trackedObjects[i].bbox,
                cameraMotion);

            cv::Point2f center(
                trackedObjects[i].bbox.x + trackedObjects[i].bbox.width / 2.0f,
                trackedObjects[i].bbox.y + trackedObjects[i].bbox.height / 2.0f);
            trackedObjects[i].predictor.update(center, trackedObjects[i].motion.envMotion);
        }
    }

    // Create new tracks for unmatched detections
    for (size_t i = 0; i < detections.size(); i++) {
        if (!detectionMatched[i]) {
            createTrack(frame, detections[i]);
        }
    }

    // Handle lost tracks
    handleLostTracks();
}

void TrackerManager::updateTrackers(const cv::Mat& frame, const CameraMotion& cameraMotion) {
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
        } else {
            track.framesLost++;
        }
    }

    handleLostTracks();
}

int TrackerManager::getActiveCount() const {
    int count = 0;
    for (const auto& track : trackedObjects) {
        if (track.active) count++;
    }
    return count;
}

int TrackerManager::addTrack(const cv::Mat& frame, const cv::Rect& bbox,
                              const std::string& className) {
    TrackedObject track;
    track.id = nextId++;
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

    // Initialize predictor
    cv::Point2f center(bbox.x + bbox.width / 2.0f, bbox.y + bbox.height / 2.0f);
    track.predictor.init(center);

    int id = track.id;
    trackedObjects.push_back(std::move(track));
    return id;
}

void TrackerManager::removeTrack(int id) {
    auto it = std::find_if(trackedObjects.begin(), trackedObjects.end(),
                           [id](const TrackedObject& t) { return t.id == id; });
    if (it != trackedObjects.end()) {
        trackedObjects.erase(it);
    }
}

void TrackerManager::clear() {
    trackedObjects.clear();
    nextId = 0;
}

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

void TrackerManager::createTrack(const cv::Mat& frame, const Detector::Detection& detection) {
    TrackedObject track;
    track.id = nextId++;
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

    // Initialize predictor
    cv::Point2f center(detection.bbox.x + detection.bbox.width / 2.0f,
                       detection.bbox.y + detection.bbox.height / 2.0f);
    track.predictor.init(center);

    trackedObjects.push_back(std::move(track));
}

void TrackerManager::updateTrack(TrackedObject& track, const cv::Mat& frame,
                                  const cv::Rect& newBbox, const CameraMotion& cameraMotion) {
    track.prevBbox = track.bbox;
    track.bbox = newBbox;
    track.framesTracked++;

    // Reinitialize tracker with detection bbox for drift correction
    track.tracker = createTracker();
    if (track.tracker) {
        track.tracker->init(frame, newBbox);
    }

    // Update motion
    track.motion = computeObjectMotion(track.prevBbox, track.bbox, cameraMotion);

    // Update predictor
    cv::Point2f center(newBbox.x + newBbox.width / 2.0f,
                       newBbox.y + newBbox.height / 2.0f);
    track.predictor.update(center, track.motion.envMotion);
}

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

cv::Scalar TrackerManager::generateColor() const {
    static std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(64, 255);
    return cv::Scalar(dist(rng), dist(rng), dist(rng));
}
