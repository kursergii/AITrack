/**
 * @file visualization.hpp
 * @brief Visualization utilities for drawing predictions and motion indicators.
 *
 * Provides functions to visualize predicted trajectories and motion direction
 * indicators on the video frame.
 */

#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

/// Visualization constants for drawing parameters
namespace VisConst {
    // Prediction drawing
    constexpr float MIN_CONFIDENCE = 0.2f;
    constexpr int PATH_STEP = 3;
    constexpr int DOT_INTERVAL = 9;
    constexpr int DOT_RADIUS = 4;
    constexpr int MARKER_SIZE = 15;
    constexpr int MARKER_THICKNESS = 2;

    // Motion indicator
    constexpr int INDICATOR_RADIUS = 40;
    constexpr float LOG_SCALE_FACTOR = 15.0f;
    constexpr int ARROW_PADDING = 5;
    constexpr int BASE_ARROW_THICKNESS = 2;
    constexpr float ARROW_THICKNESS_SCALE = 10.0f;
    constexpr int MAX_ARROW_THICKNESS = 5;
    constexpr double ARROW_TIP_RATIO = 0.3;
    constexpr int STILL_DOT_RADIUS = 5;

    // Colors
    const cv::Scalar INDICATOR_BG(50, 50, 50);
    const cv::Scalar INDICATOR_BORDER(200, 200, 200);
    const cv::Scalar MARKER_COLOR(255, 100, 0);
    const cv::Scalar STILL_COLOR(0, 255, 0);
}

/**
 * @brief Draw predicted trajectory path on frame.
 * @param frame Frame to draw on
 * @param currentCenter Current object center position
 * @param predictedPath Vector of predicted future positions
 * @param confidence Prediction confidence (0-1), affects opacity and thickness
 *
 * Draws a fading line from current position through predicted positions,
 * with periodic dots and a final cross marker at the predicted endpoint.
 */
void drawPrediction(cv::Mat& frame, const cv::Point2f& currentCenter,
                    const std::vector<cv::Point2f>& predictedPath, float confidence);

/**
 * @brief Draw motion direction indicator (compass-style arrow).
 * @param frame Frame to draw on
 * @param direction Motion direction vector
 * @param magnitude Motion magnitude in pixels/frame
 * @param center Position to draw the indicator
 * @param color Arrow color
 * @param threshold Minimum magnitude to show arrow (below shows dot for "still")
 *
 * Draws a circular indicator with an arrow showing motion direction.
 * Arrow length uses log scale to visualize both slow and fast motion clearly.
 */
void drawMotionIndicator(cv::Mat& frame, const cv::Point2f& direction, float magnitude,
                         cv::Point center, cv::Scalar color, float threshold = 1.5f);
