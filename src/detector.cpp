/**
 * @file detector.cpp
 * @brief Implementation of YOLO object detector using OpenCV DNN.
 *
 * Supports YOLOv5, YOLOv8, and YOLOv11 ONNX models. Auto-detects output format
 * and handles letterbox preprocessing for proper aspect ratio preservation.
 */

#include "detector.hpp"
#include <fstream>
#include <iostream>
#include <opencv2/core/ocl.hpp>

// ============================================================================
// Constructor / Destructor
// ============================================================================

Detector::Detector()
    : loaded(false), confidenceThreshold(0.5f), nmsThreshold(0.45f), inputSize(640),
      backend(cv::dnn::DNN_BACKEND_OPENCV), target(cv::dnn::DNN_TARGET_CPU),
      letterboxScale(1.0f), letterboxPadX(0), letterboxPadY(0) {
}

Detector::~Detector() {
}

// ============================================================================
// Backend Configuration
// ============================================================================

/// Set compute backend manually (CUDA, OpenCL, or CPU)
void Detector::setBackend(cv::dnn::Backend b, cv::dnn::Target t) {
    backend = b;
    target = t;
    if (loaded) {
        net.setPreferableBackend(backend);
        net.setPreferableTarget(target);
    }
}

/**
 * @brief Auto-detect and enable GPU acceleration.
 * @return true if GPU enabled, false if falling back to CPU
 *
 * Priority: CUDA > OpenCL > CPU
 */
bool Detector::tryEnableGPU() {
    // Check if CUDA is available (OpenCV must be built with CUDA support)
    int cudaDevices = cv::cuda::getCudaEnabledDeviceCount();
    if (cudaDevices > 0) {
        backend = cv::dnn::DNN_BACKEND_CUDA;
        target = cv::dnn::DNN_TARGET_CUDA;
        if (loaded) {
            net.setPreferableBackend(backend);
            net.setPreferableTarget(target);
        }
        std::cout << "CUDA backend enabled (" << cudaDevices << " device(s))" << std::endl;
        return true;
    }

    // Check if OpenCL is available
    if (cv::ocl::haveOpenCL()) {
        backend = cv::dnn::DNN_BACKEND_OPENCV;
        target = cv::dnn::DNN_TARGET_OPENCL;
        if (loaded) {
            net.setPreferableBackend(backend);
            net.setPreferableTarget(target);
        }
        std::cout << "OpenCL backend enabled" << std::endl;
        return true;
    }

    // Fall back to CPU
    backend = cv::dnn::DNN_BACKEND_OPENCV;
    target = cv::dnn::DNN_TARGET_CPU;
    if (loaded) {
        net.setPreferableBackend(backend);
        net.setPreferableTarget(target);
    }
    std::cout << "Using CPU backend" << std::endl;
    return false;
}

std::string Detector::getBackendName() const {
    if (target == cv::dnn::DNN_TARGET_CUDA) return "CUDA";
    if (target == cv::dnn::DNN_TARGET_OPENCL) return "OpenCL";
    return "CPU";
}

// ============================================================================
// Model Loading
// ============================================================================

/// Load YOLO ONNX model and optionally class names from file
bool Detector::load(const std::string& modelPath, const std::string& classesPath) {
    try {
        net = cv::dnn::readNetFromONNX(modelPath);
        net.setPreferableBackend(backend);
        net.setPreferableTarget(target);

        // Load class names if provided
        if (!classesPath.empty()) {
            std::ifstream ifs(classesPath);
            if (ifs.is_open()) {
                std::string line;
                while (std::getline(ifs, line)) {
                    if (!line.empty()) {
                        classNames.push_back(line);
                    }
                }
            }
        }

        // Default COCO classes if not provided
        if (classNames.empty()) {
            classNames = {
                "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
                "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
                "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
                "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
                "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
                "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
                "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
                "toothbrush"
            };
        }

        loaded = true;
        std::cout << "YOLO model loaded: " << modelPath << std::endl;
        std::cout << "Classes: " << classNames.size() << std::endl;
        return true;

    } catch (const cv::Exception& e) {
        std::cerr << "Error loading YOLO model: " << e.what() << std::endl;
        loaded = false;
        return false;
    }
}

// ============================================================================
// Detection Pipeline
// ============================================================================

/**
 * @brief Preprocess frame for YOLO input.
 *
 * Letterbox resize: scales image to fit inputSize while maintaining aspect ratio,
 * then pads with gray (114) to create a square image. Saves scale/padding info
 * for coordinate conversion in postprocess.
 */
cv::Mat Detector::preprocess(const cv::Mat& frame) {
    // Letterbox resize: maintain aspect ratio with padding
    int imgWidth = frame.cols;
    int imgHeight = frame.rows;

    // Calculate scale to fit in inputSize while maintaining aspect ratio
    float scaleX = static_cast<float>(inputSize) / imgWidth;
    float scaleY = static_cast<float>(inputSize) / imgHeight;
    letterboxScale = std::min(scaleX, scaleY);

    int newWidth = static_cast<int>(imgWidth * letterboxScale);
    int newHeight = static_cast<int>(imgHeight * letterboxScale);

    // Calculate padding (center the image)
    letterboxPadX = (inputSize - newWidth) / 2;
    letterboxPadY = (inputSize - newHeight) / 2;

    // Resize image
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);

    // Create letterbox image with gray padding
    cv::Mat letterboxed(inputSize, inputSize, CV_8UC3,
                        cv::Scalar(LETTERBOX_PAD_VALUE, LETTERBOX_PAD_VALUE, LETTERBOX_PAD_VALUE));
    resized.copyTo(letterboxed(cv::Rect(letterboxPadX, letterboxPadY, newWidth, newHeight)));

    // Convert to blob: normalize, swap RB, CHW format
    cv::Mat blob;
    cv::dnn::blobFromImage(letterboxed, blob, NORMALIZATION_FACTOR, cv::Size(inputSize, inputSize),
                           cv::Scalar(), true, false);
    return blob;
}

/// Run detection on a frame (preprocess -> forward -> postprocess)
std::vector<Detector::Detection> Detector::detect(const cv::Mat& frame) {
    std::vector<Detection> detections;

    if (!loaded || frame.empty()) {
        return detections;
    }

    // Preprocess
    cv::Mat blob = preprocess(frame);
    net.setInput(blob);

    // Forward pass
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    // Postprocess
    return postprocess(frame, outputs);
}

/**
 * @brief Postprocess YOLO output to extract detections.
 *
 * Auto-detects YOLOv5 vs v8/v11 format based on output tensor shape:
 * - YOLOv8/v11: [1, 84, 8400] - needs transpose, no objectness score
 * - YOLOv5: [1, 25200, 85] - has objectness score at index 4
 *
 * Applies confidence thresholding, coordinate conversion from letterbox
 * to original image space, and Non-Maximum Suppression.
 */
std::vector<Detector::Detection> Detector::postprocess(const cv::Mat& frame,
                                                        const std::vector<cv::Mat>& outputs) {
    std::vector<Detection> detections;
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    int imgWidth = frame.cols;
    int imgHeight = frame.rows;

    // YOLOv5/v8/v11 output format:
    // - YOLOv8/v11: [1, 84, 8400] where 84 = 4 coords + 80 classes, 8400 = predictions
    // - YOLOv5: [1, 25200, 85] where 85 = 4 coords + 1 objectness + 80 classes

    cv::Mat output = outputs[0];

    // Check output dimensions to determine format
    int dim1 = output.size[1];
    int dim2 = output.size[2];

    // YOLOv8/v11 format: [1, 84, 8400] - features x predictions
    // YOLOv5 format: [1, 25200, 85] - predictions x features
    bool isV8Format = (dim2 > dim1);

    int rows, cols;
    cv::Mat output2d;

    if (isV8Format) {
        // YOLOv8/v11: [1, features, predictions] -> need to transpose
        // Data layout: feature0[pred0..pred8399], feature1[pred0..pred8399], ...
        // Create Mat matching data layout: (features rows, predictions cols)
        cv::Mat temp(dim1, dim2, CV_32F, output.data);
        // Transpose to get: (predictions rows, features cols)
        cv::transpose(temp, output2d);
        rows = output2d.rows;  // 8400 predictions
        cols = output2d.cols;  // 84 features
    } else {
        // YOLOv5: [1, predictions, features] - already in correct layout
        output2d = cv::Mat(dim1, dim2, CV_32F, output.data);
        rows = dim1;
        cols = dim2;
    }

    float* data = reinterpret_cast<float*>(output2d.data);

    for (int i = 0; i < rows; i++) {
        float* row = data + i * cols;

        if (isV8Format) {
            // YOLOv8/v11: no objectness score, class scores start at index 4
            float* classScores = row + 4;
            int numClasses = cols - 4;

            cv::Mat scores(1, numClasses, CV_32FC1, classScores);
            cv::Point classIdPoint;
            double maxClassScore;
            cv::minMaxLoc(scores, nullptr, &maxClassScore, nullptr, &classIdPoint);

            if (maxClassScore > confidenceThreshold) {
                float cx = row[0];
                float cy = row[1];
                float w = row[2];
                float h = row[3];

                // Convert from letterbox coordinates to original image coordinates
                // 1. Remove padding offset
                // 2. Scale back to original image size
                int left = static_cast<int>((cx - w / 2 - letterboxPadX) / letterboxScale);
                int top = static_cast<int>((cy - h / 2 - letterboxPadY) / letterboxScale);
                int width = static_cast<int>(w / letterboxScale);
                int height = static_cast<int>(h / letterboxScale);

                // Clamp to image boundaries
                left = std::max(0, std::min(left, imgWidth - 1));
                top = std::max(0, std::min(top, imgHeight - 1));
                width = std::min(width, imgWidth - left);
                height = std::min(height, imgHeight - top);

                if (width > 0 && height > 0) {
                    boxes.push_back(cv::Rect(left, top, width, height));
                    confidences.push_back(static_cast<float>(maxClassScore));
                    classIds.push_back(classIdPoint.x);
                }
            }
        } else {
            // YOLOv5: objectness score at index 4, class scores start at index 5
            float objectness = row[4];

            if (objectness > confidenceThreshold) {
                float* classScores = row + 5;
                int numClasses = cols - 5;

                cv::Mat scores(1, numClasses, CV_32FC1, classScores);
                cv::Point classIdPoint;
                double maxClassScore;
                cv::minMaxLoc(scores, nullptr, &maxClassScore, nullptr, &classIdPoint);

                float confidence = objectness * static_cast<float>(maxClassScore);

                if (confidence > confidenceThreshold) {
                    float cx = row[0];
                    float cy = row[1];
                    float w = row[2];
                    float h = row[3];

                    // Convert from letterbox coordinates to original image coordinates
                    int left = static_cast<int>((cx - w / 2 - letterboxPadX) / letterboxScale);
                    int top = static_cast<int>((cy - h / 2 - letterboxPadY) / letterboxScale);
                    int width = static_cast<int>(w / letterboxScale);
                    int height = static_cast<int>(h / letterboxScale);

                    // Clamp to image boundaries
                    left = std::max(0, std::min(left, imgWidth - 1));
                    top = std::max(0, std::min(top, imgHeight - 1));
                    width = std::min(width, imgWidth - left);
                    height = std::min(height, imgHeight - top);

                    if (width > 0 && height > 0) {
                        boxes.push_back(cv::Rect(left, top, width, height));
                        confidences.push_back(confidence);
                        classIds.push_back(classIdPoint.x);
                    }
                }
            }
        }
    }

    // Apply NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confidenceThreshold, nmsThreshold, indices);

    // Collect NMS survivors
    std::vector<Detection> nmsDetections;
    for (int idx : indices) {
        Detection det;
        det.bbox = boxes[idx];
        det.classId = classIds[idx];
        det.confidence = confidences[idx];

        if (det.classId >= 0 && det.classId < static_cast<int>(classNames.size())) {
            det.className = classNames[det.classId];
        } else {
            det.className = "class_" + std::to_string(det.classId);
        }

        nmsDetections.push_back(det);
    }

    // Post-NMS merge: suppress nearby same-class detections using distance.
    // If centers are closer than the size of the larger box, keep only the
    // higher-confidence one.
    std::vector<bool> suppressed(nmsDetections.size(), false);
    for (size_t i = 0; i < nmsDetections.size(); i++) {
        if (suppressed[i]) continue;
        for (size_t j = i + 1; j < nmsDetections.size(); j++) {
            if (suppressed[j]) continue;
            if (nmsDetections[i].classId != nmsDetections[j].classId) continue;

            cv::Point2f ci(nmsDetections[i].bbox.x + nmsDetections[i].bbox.width / 2.0f,
                           nmsDetections[i].bbox.y + nmsDetections[i].bbox.height / 2.0f);
            cv::Point2f cj(nmsDetections[j].bbox.x + nmsDetections[j].bbox.width / 2.0f,
                           nmsDetections[j].bbox.y + nmsDetections[j].bbox.height / 2.0f);

            float dist = cv::norm(ci - cj);
            float maxDim = static_cast<float>(std::max({
                nmsDetections[i].bbox.width, nmsDetections[i].bbox.height,
                nmsDetections[j].bbox.width, nmsDetections[j].bbox.height
            }));

            if (dist < maxDim * 2.0f) {
                if (nmsDetections[i].confidence >= nmsDetections[j].confidence)
                    suppressed[j] = true;
                else
                    suppressed[i] = true;
            }
        }
    }

    for (size_t i = 0; i < nmsDetections.size(); i++) {
        if (!suppressed[i])
            detections.push_back(nmsDetections[i]);
    }

    return detections;
}
