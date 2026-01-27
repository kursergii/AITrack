#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

// YOLO-based object detector (to be implemented)
class Detector {
public:
    struct Detection {
        cv::Rect bbox;
        int classId;
        float confidence;
        std::string className;
    };

    Detector();
    ~Detector();

    bool load(const std::string& modelPath, const std::string& configPath = "");
    std::vector<Detection> detect(const cv::Mat& frame);

private:
    // TODO: Add YOLO net and configuration
};
