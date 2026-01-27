#pragma once

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

// YOLO-based object detector (supports YOLOv5/v8/v11 ONNX models)
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

    // Load YOLO ONNX model
    bool load(const std::string& modelPath, const std::string& classesPath = "");

    // Run detection on frame
    std::vector<Detection> detect(const cv::Mat& frame);

    // Configuration
    void setConfidenceThreshold(float thresh) { confidenceThreshold = thresh; }
    void setNmsThreshold(float thresh) { nmsThreshold = thresh; }
    void setInputSize(int size) { inputSize = size; }

    float getConfidenceThreshold() const { return confidenceThreshold; }
    bool isLoaded() const { return loaded; }

private:
    cv::dnn::Net net;
    std::vector<std::string> classNames;
    bool loaded;

    // Parameters
    float confidenceThreshold;
    float nmsThreshold;
    int inputSize;

    // Letterbox state (for coordinate conversion)
    float letterboxScale;
    int letterboxPadX;
    int letterboxPadY;

    // Preprocessing with letterbox (maintains aspect ratio)
    cv::Mat preprocess(const cv::Mat& frame);

    // Postprocessing (handles YOLOv5/v8/v11 output format)
    std::vector<Detection> postprocess(const cv::Mat& frame, const std::vector<cv::Mat>& outputs);
};
