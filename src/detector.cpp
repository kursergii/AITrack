#include "detector.hpp"
#include <fstream>
#include <iostream>

Detector::Detector()
    : loaded(false), confidenceThreshold(0.5f), nmsThreshold(0.45f), inputSize(640),
      letterboxScale(1.0f), letterboxPadX(0), letterboxPadY(0) {
}

Detector::~Detector() {
}

bool Detector::load(const std::string& modelPath, const std::string& classesPath) {
    try {
        net = cv::dnn::readNetFromONNX(modelPath);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

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

    // Create letterbox image with gray padding (114 is standard YOLO padding)
    cv::Mat letterboxed(inputSize, inputSize, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(letterboxed(cv::Rect(letterboxPadX, letterboxPadY, newWidth, newHeight)));

    // Convert to blob: normalize, swap RB, CHW format
    cv::Mat blob;
    cv::dnn::blobFromImage(letterboxed, blob, 1.0/255.0, cv::Size(inputSize, inputSize),
                           cv::Scalar(), true, false);
    return blob;
}

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

        detections.push_back(det);
    }

    return detections;
}
