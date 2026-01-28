/**
 * @file visualization.cpp
 * @brief Visualization utilities for trajectory prediction and motion indicators.
 */

#include "visualization.hpp"
#include <algorithm>
#include <cmath>

/**
 * @brief Draw predicted trajectory path on frame.
 *
 * Draws a fading line from current position through predicted positions.
 * Line thickness and opacity decrease with distance. Periodic dots mark
 * the path, and a cross marker shows the final predicted position.
 */
void drawPrediction(cv::Mat& frame, const cv::Point2f& currentCenter,
                    const std::vector<cv::Point2f>& predictedPath, float confidence) {
    if (predictedPath.empty() || confidence < VisConst::MIN_CONFIDENCE) return;

    cv::Point2f prevPoint = currentCenter;

    for (size_t i = 0; i < predictedPath.size(); i += VisConst::PATH_STEP) {
        float alpha = 1.0f - (static_cast<float>(i) / predictedPath.size());
        int thickness = std::max(1, static_cast<int>(VisConst::PATH_STEP * alpha * confidence));

        // Color fades from cyan to blue
        cv::Scalar color(255, static_cast<int>(255 * alpha), 0);

        cv::Point2f point = predictedPath[i];

        // Check bounds
        if (point.x >= 0 && point.x < frame.cols && point.y >= 0 && point.y < frame.rows) {
            cv::line(frame, cv::Point(static_cast<int>(prevPoint.x), static_cast<int>(prevPoint.y)),
                     cv::Point(static_cast<int>(point.x), static_cast<int>(point.y)),
                     color, thickness, cv::LINE_AA);

            // Draw prediction dots
            if (i % VisConst::DOT_INTERVAL == 0) {
                int dotRadius = std::max(1, static_cast<int>(VisConst::DOT_RADIUS * alpha * confidence));
                cv::circle(frame, cv::Point(static_cast<int>(point.x), static_cast<int>(point.y)),
                           dotRadius, color, -1);
            }
        }
        prevPoint = point;
    }

    // Draw final predicted position marker
    if (!predictedPath.empty()) {
        cv::Point2f finalPos = predictedPath.back();
        if (finalPos.x >= 0 && finalPos.x < frame.cols && finalPos.y >= 0 && finalPos.y < frame.rows) {
            cv::drawMarker(frame, cv::Point(static_cast<int>(finalPos.x), static_cast<int>(finalPos.y)),
                           VisConst::MARKER_COLOR, cv::MARKER_CROSS,
                           VisConst::MARKER_SIZE, VisConst::MARKER_THICKNESS);
        }
    }
}

/**
 * @brief Draw compass-style motion direction indicator.
 *
 * Shows a circular indicator with:
 * - Arrow pointing in motion direction (length scaled logarithmically)
 * - Arrow thickness proportional to speed
 * - Green dot when stationary (below threshold)
 * - Magnitude value displayed below
 */
void drawMotionIndicator(cv::Mat& frame, const cv::Point2f& direction, float magnitude,
                         cv::Point center, cv::Scalar color, float threshold) {
    constexpr int radius = VisConst::INDICATOR_RADIUS;

    // Draw circle background
    cv::circle(frame, center, radius, VisConst::INDICATOR_BG, -1);
    cv::circle(frame, center, radius, VisConst::INDICATOR_BORDER, 2);

    // Draw arrow if moving
    if (magnitude > threshold) {
        // Scale arrow: use log scale for better visualization of both slow and fast motion
        // log(1 + mag) gives ~0.7 for mag=1, ~2.4 for mag=10, ~4.6 for mag=100
        float logMag = std::log(1.0f + magnitude);
        float scale = std::min(logMag * VisConst::LOG_SCALE_FACTOR,
                               static_cast<float>(radius - VisConst::ARROW_PADDING));

        cv::Point2f dir = direction / magnitude;
        cv::Point arrowEnd(center.x + static_cast<int>(dir.x * scale),
                          center.y + static_cast<int>(dir.y * scale));

        // Thicker arrow for faster movement (ensure at least 1)
        int thickness = std::max(1, std::min(VisConst::BASE_ARROW_THICKNESS +
                                 static_cast<int>(magnitude / VisConst::ARROW_THICKNESS_SCALE),
                                 VisConst::MAX_ARROW_THICKNESS));
        cv::arrowedLine(frame, center, arrowEnd, color, thickness, cv::LINE_AA, 0,
                        VisConst::ARROW_TIP_RATIO);

        // Show magnitude value
        std::string magText = std::to_string(static_cast<int>(magnitude));
        cv::putText(frame, magText, cv::Point(center.x - 10, center.y + radius + 15),
                   cv::FONT_HERSHEY_SIMPLEX, 0.35, color, 1);
    } else {
        // Draw dot for still
        cv::circle(frame, center, VisConst::STILL_DOT_RADIUS, VisConst::STILL_COLOR, -1);
    }
}
