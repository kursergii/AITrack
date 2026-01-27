#pragma once

#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

class NanoTracker {
public:
    NanoTracker();
    ~NanoTracker();

    bool load(const std::string& backbonePath, const std::string& neckheadPath);
    void init(const cv::Mat& frame, const cv::Rect& roi);
    bool update(const cv::Mat& frame, cv::Rect& bbox);
    void reset();

    bool isInitialized() const { return initialized; }
    bool isTracking() const { return tracking; }
    cv::Rect getBbox() const { return bbox; }

private:
    cv::Ptr<cv::TrackerNano> tracker;
    cv::TrackerNano::Params params;
    cv::Rect bbox;
    bool initialized;
    bool tracking;
};
