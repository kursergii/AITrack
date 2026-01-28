/**
 * @file kalman_predictor.hpp
 * @brief Kalman filter-based motion predictor for trajectory forecasting.
 *
 * Uses a constant velocity model with 4 state variables [x, y, vx, vy].
 * Maintains velocity history for stable predictions and uses median filtering
 * to reduce jitter from noisy measurements.
 */

#pragma once

#include <deque>
#include <vector>
#include <opencv2/opencv.hpp>

/**
 * @class KalmanPredictor
 * @brief Kalman filter-based motion predictor with velocity smoothing.
 *
 * State vector: [x, y, vx, vy] - position and velocity (constant velocity model).
 * Uses exponential moving average and median filtering for stable velocity estimation.
 * Provides confidence scores based on velocity consistency and update count.
 */
class KalmanPredictor {
public:
    // Existing constants
    static const int PREDICTION_FRAMES = 30;
    static const int VELOCITY_HISTORY_SIZE = 15;

    // Kalman filter noise parameters
    static constexpr float PROCESS_NOISE_POSITION = 0.05f;
    static constexpr float PROCESS_NOISE_VELOCITY = 0.5f;
    static constexpr float MEASUREMENT_NOISE = 10.0f;

    // Velocity smoothing
    static constexpr float EMA_ALPHA = 0.25f;                 // Exponential moving average factor
    static constexpr int MIN_HISTORY_FOR_MEDIAN = 5;          // Min samples for median velocity

    // Prediction thresholds
    static constexpr int MIN_UPDATES_FOR_PREDICTION = 10;     // Frames before prediction starts
    static constexpr int MIN_UPDATES_FOR_CONFIDENCE = 12;     // Frames before confidence > 0
    static constexpr float STABLE_VELOCITY_WEIGHT = 0.7f;     // Blend weight for stable velocity
    static constexpr float SMOOTHED_VELOCITY_WEIGHT = 0.3f;   // Blend weight for smoothed velocity
    static constexpr float MIN_SPEED_FOR_PREDICTION = 1.0f;   // Pixels/frame
    static constexpr float MIN_CONSISTENCY = 0.3f;            // Minimum velocity consistency
    static constexpr float VELOCITY_DECAY = 0.97f;            // Per-frame decay in prediction

    // Confidence calculation
    static constexpr float MIN_SPEED_FOR_CONFIDENCE = 1.0f;
    static constexpr float SPEED_FACTOR_SCALE = 8.0f;
    static constexpr float CONFIDENCE_BASE = 0.4f;
    static constexpr float CONFIDENCE_SPEED_WEIGHT = 0.6f;

    KalmanPredictor();

    void init(const cv::Point2f& pos);
    void reset();
    void update(const cv::Point2f& measuredPos, const cv::Point2f& envVelocity);

    // Get current state
    cv::Point2f getPosition() const;
    cv::Point2f getVelocity() const;

    // Predict future positions
    std::vector<cv::Point2f> predictPath(int numFrames) const;

    // Compute how consistent the velocity has been (0-1)
    float computeVelocityConsistency() const;
    float getConfidence() const;

private:
    // Compute median velocity from history (more robust to outliers)
    cv::Point2f computeMedianVelocity() const;

    cv::KalmanFilter kf;
    bool initialized;
    int updateCount;

    // Velocity history for stable prediction
    std::deque<cv::Point2f> velocityHistory;
    cv::Point2f smoothedVelocity;
    cv::Point2f stableVelocity;  // Extra-stable velocity for prediction
    cv::Point2f prevPosition;
};
