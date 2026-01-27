#include "nanotracker.hpp"
#include <iostream>

NanoTracker::NanoTracker() : initialized(false), tracking(false) {
}

NanoTracker::~NanoTracker() {
}

bool NanoTracker::load(const std::string& backbonePath, const std::string& neckheadPath) {
    params.backbone = backbonePath;
    params.neckhead = neckheadPath;

    try {
        tracker = cv::TrackerNano::create(params);
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "Error creating NanoTrack tracker: " << e.what() << std::endl;
        std::cerr << "Make sure model files exist at:" << std::endl;
        std::cerr << "  Backbone: " << backbonePath << std::endl;
        std::cerr << "  Neckhead: " << neckheadPath << std::endl;
        return false;
    }
}

void NanoTracker::init(const cv::Mat& frame, const cv::Rect& roi) {
    if (!tracker) {
        std::cerr << "Tracker not loaded. Call load() first." << std::endl;
        return;
    }

    // Recreate tracker for re-initialization
    tracker = cv::TrackerNano::create(params);
    tracker->init(frame, roi);
    bbox = roi;
    initialized = true;
    tracking = true;
}

bool NanoTracker::update(const cv::Mat& frame, cv::Rect& outBbox) {
    if (!initialized || !tracker) {
        return false;
    }

    tracking = tracker->update(frame, bbox);
    outBbox = bbox;
    return tracking;
}

void NanoTracker::reset() {
    initialized = false;
    tracking = false;
    bbox = cv::Rect();
}
