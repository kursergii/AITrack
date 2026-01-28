/**
 * @file detector.hpp
 * @brief YOLO object detector wrapper for OpenCV DNN.
 *
 * Supports YOLOv5, YOLOv8, and YOLOv11 ONNX models with automatic format detection.
 * Handles letterbox preprocessing to maintain aspect ratio and supports CUDA,
 * OpenCL, and CPU backends for inference.
 */

#pragma once

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

/**
 * @class Detector
 * @brief YOLO-based object detector supporting YOLOv5/v8/v11 ONNX models.
 *
 * Preprocessing: Letterbox resize maintains aspect ratio with gray padding (114).
 * Postprocessing: Auto-detects output format (v5 vs v8/v11) and applies NMS.
 */
class Detector {
public:
    // Preprocessing constants
    static constexpr int LETTERBOX_PAD_VALUE = 114;      // Standard YOLO padding (gray)
    static constexpr double NORMALIZATION_FACTOR = 1.0 / 255.0;

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

    // Backend selection (call before load() or use tryEnableGPU() after load())
    void setBackend(cv::dnn::Backend backend, cv::dnn::Target target);
    bool tryEnableGPU();  // Auto-detect and enable CUDA/OpenCL if available

    float getConfidenceThreshold() const { return confidenceThreshold; }
    bool isLoaded() const { return loaded; }
    std::string getBackendName() const;

private:
    cv::dnn::Net net;
    std::vector<std::string> classNames;
    bool loaded;

    // Parameters
    float confidenceThreshold;
    float nmsThreshold;
    int inputSize;

    // Backend
    cv::dnn::Backend backend;
    cv::dnn::Target target;

    // Letterbox state (for coordinate conversion)
    float letterboxScale;
    int letterboxPadX;
    int letterboxPadY;

    // Preprocessing with letterbox (maintains aspect ratio)
    cv::Mat preprocess(const cv::Mat& frame);

    // Postprocessing (handles YOLOv5/v8/v11 output format)
    std::vector<Detection> postprocess(const cv::Mat& frame, const std::vector<cv::Mat>& outputs);
};
