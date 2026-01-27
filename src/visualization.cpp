#include "visualization.hpp"
#include <algorithm>
#include <cmath>

void drawPrediction(cv::Mat& frame, const cv::Point2f& currentCenter,
                    const std::vector<cv::Point2f>& predictedPath, float confidence) {
    if (predictedPath.empty() || confidence < 0.2f) return;

    cv::Point2f prevPoint = currentCenter;

    for (size_t i = 0; i < predictedPath.size(); i += 3) {
        float alpha = 1.0f - (static_cast<float>(i) / predictedPath.size());
        int thickness = std::max(1, static_cast<int>(3 * alpha * confidence));

        // Color fades from cyan to blue
        cv::Scalar color(255, static_cast<int>(255 * alpha), 0);

        cv::Point2f point = predictedPath[i];

        // Check bounds
        if (point.x >= 0 && point.x < frame.cols && point.y >= 0 && point.y < frame.rows) {
            cv::line(frame, cv::Point(static_cast<int>(prevPoint.x), static_cast<int>(prevPoint.y)),
                     cv::Point(static_cast<int>(point.x), static_cast<int>(point.y)),
                     color, thickness, cv::LINE_AA);

            // Draw prediction dots
            if (i % 9 == 0) {
                cv::circle(frame, cv::Point(static_cast<int>(point.x), static_cast<int>(point.y)),
                           static_cast<int>(4 * alpha * confidence), color, -1);
            }
        }
        prevPoint = point;
    }

    // Draw final predicted position marker
    if (!predictedPath.empty()) {
        cv::Point2f finalPos = predictedPath.back();
        if (finalPos.x >= 0 && finalPos.x < frame.cols && finalPos.y >= 0 && finalPos.y < frame.rows) {
            cv::drawMarker(frame, cv::Point(static_cast<int>(finalPos.x), static_cast<int>(finalPos.y)),
                           cv::Scalar(255, 100, 0), cv::MARKER_CROSS, 15, 2);
        }
    }
}

void drawMotionIndicator(cv::Mat& frame, const cv::Point2f& direction, float magnitude,
                         cv::Point center, cv::Scalar color, float threshold) {
    int radius = 40;

    // Draw circle background
    cv::circle(frame, center, radius, cv::Scalar(50, 50, 50), -1);
    cv::circle(frame, center, radius, cv::Scalar(200, 200, 200), 2);

    // Draw arrow if moving
    if (magnitude > threshold) {
        float scale = std::min(magnitude * 3.0f, static_cast<float>(radius - 5));
        cv::Point2f dir = direction / magnitude;
        cv::Point arrowEnd(center.x + static_cast<int>(dir.x * scale),
                          center.y + static_cast<int>(dir.y * scale));
        cv::arrowedLine(frame, center, arrowEnd, color, 3, cv::LINE_AA, 0, 0.3);
    } else {
        // Draw dot for still
        cv::circle(frame, center, 5, cv::Scalar(0, 255, 0), -1);
    }
}
