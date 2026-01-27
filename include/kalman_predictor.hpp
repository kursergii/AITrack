#pragma once

#include <deque>
#include <vector>
#include <opencv2/opencv.hpp>

// Kalman filter based motion predictor with improved stability
// State: [x, y, vx, vy] - position and velocity (constant velocity model)
class KalmanPredictor {
public:
    static const int PREDICTION_FRAMES = 30;
    static const int VELOCITY_HISTORY_SIZE = 15;

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
