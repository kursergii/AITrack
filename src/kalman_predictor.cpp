/**
 * @file kalman_predictor.cpp
 * @brief Kalman filter-based motion predictor implementation.
 *
 * Uses a constant velocity model for position prediction with additional
 * velocity smoothing (EMA + median) for stable trajectory forecasting.
 */

#include "kalman_predictor.hpp"
#include <algorithm>
#include <cmath>

// ============================================================================
// Constructor - Kalman Filter Setup
// ============================================================================

KalmanPredictor::KalmanPredictor()
    : initialized(false), updateCount(0),
      smoothedVelocity(0, 0), stableVelocity(0, 0), prevPosition(0, 0) {
    // 4 state variables (x, y, vx, vy), 2 measurements (x, y)
    kf = cv::KalmanFilter(4, 2, 0);

    // Transition matrix (constant velocity model)
    kf.transitionMatrix = (cv::Mat_<float>(4, 4) <<
        1, 0, 1, 0,
        0, 1, 0, 1,
        0, 0, 1, 0,
        0, 0, 0, 1);

    // Measurement matrix (we only measure x, y)
    kf.measurementMatrix = (cv::Mat_<float>(2, 4) <<
        1, 0, 0, 0,
        0, 1, 0, 0);

    // Process noise - lower values = smoother but slower response
    kf.processNoiseCov = (cv::Mat_<float>(4, 4) <<
        PROCESS_NOISE_POSITION, 0,                     0,                     0,
        0,                      PROCESS_NOISE_POSITION, 0,                     0,
        0,                      0,                      PROCESS_NOISE_VELOCITY, 0,
        0,                      0,                      0,                      PROCESS_NOISE_VELOCITY);

    // Measurement noise - higher = trust measurements less
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(MEASUREMENT_NOISE));

    // Initial error covariance
    cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1));
}

// ============================================================================
// Initialization / Reset
// ============================================================================

/// Initialize Kalman filter with starting position (zero velocity)
void KalmanPredictor::init(const cv::Point2f& pos) {
    kf.statePost = (cv::Mat_<float>(4, 1) << pos.x, pos.y, 0, 0);
    prevPosition = pos;
    smoothedVelocity = cv::Point2f(0, 0);
    stableVelocity = cv::Point2f(0, 0);
    velocityHistory.clear();
    initialized = true;
    updateCount = 0;
}

void KalmanPredictor::reset() {
    initialized = false;
    updateCount = 0;
    smoothedVelocity = cv::Point2f(0, 0);
    stableVelocity = cv::Point2f(0, 0);
    velocityHistory.clear();
    cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1));
}

// ============================================================================
// Update
// ============================================================================

/**
 * @brief Update Kalman filter with new measurement.
 *
 * Runs predict-correct cycle, then updates velocity history and computes
 * smoothed/stable velocity for trajectory prediction.
 */
void KalmanPredictor::update(const cv::Point2f& measuredPos, const cv::Point2f& envVelocity) {
    if (!initialized) {
        init(measuredPos);
        return;
    }

    // Predict step
    kf.predict();

    // Correct with measurement
    cv::Mat measurement = (cv::Mat_<float>(2, 1) << measuredPos.x, measuredPos.y);
    kf.correct(measurement);

    // Add to velocity history
    velocityHistory.push_back(envVelocity);
    if (velocityHistory.size() > VELOCITY_HISTORY_SIZE) {
        velocityHistory.pop_front();
    }

    // Compute smoothed velocity (exponential moving average)
    smoothedVelocity = smoothedVelocity * (1.0f - EMA_ALPHA) + envVelocity * EMA_ALPHA;

    // Compute extra-stable velocity using median of history
    if (velocityHistory.size() >= MIN_HISTORY_FOR_MEDIAN) {
        stableVelocity = computeMedianVelocity();
    } else {
        stableVelocity = smoothedVelocity;
    }

    prevPosition = measuredPos;
    updateCount++;
}

// ============================================================================
// Velocity Computation
// ============================================================================

/// Compute median velocity from history (robust to outliers)
cv::Point2f KalmanPredictor::computeMedianVelocity() const {
    if (velocityHistory.empty()) return cv::Point2f(0, 0);

    std::vector<float> vx, vy;
    for (const auto& v : velocityHistory) {
        vx.push_back(v.x);
        vy.push_back(v.y);
    }

    std::sort(vx.begin(), vx.end());
    std::sort(vy.begin(), vy.end());

    size_t mid = vx.size() / 2;
    float medX = (vx.size() % 2 == 0) ? (vx[mid - 1] + vx[mid]) / 2.0f : vx[mid];
    float medY = (vy.size() % 2 == 0) ? (vy[mid - 1] + vy[mid]) / 2.0f : vy[mid];

    return cv::Point2f(medX, medY);
}

cv::Point2f KalmanPredictor::getPosition() const {
    return cv::Point2f(kf.statePost.at<float>(0), kf.statePost.at<float>(1));
}

cv::Point2f KalmanPredictor::getVelocity() const {
    return stableVelocity;
}

// ============================================================================
// Prediction
// ============================================================================

/**
 * @brief Predict future positions.
 *
 * Uses blended stable/smoothed velocity with per-frame decay to predict
 * trajectory. Returns empty if insufficient updates or inconsistent velocity.
 */
std::vector<cv::Point2f> KalmanPredictor::predictPath(int numFrames) const {
    std::vector<cv::Point2f> path;

    if (!initialized || updateCount < MIN_UPDATES_FOR_PREDICTION) return path;

    // Use Kalman position and stable velocity for prediction
    cv::Point2f pos = getPosition();

    // Blend smoothed and stable velocity for best results
    cv::Point2f vel = stableVelocity * STABLE_VELOCITY_WEIGHT +
                      smoothedVelocity * SMOOTHED_VELOCITY_WEIGHT;

    // Only predict if there's meaningful and consistent velocity
    float speed = std::sqrt(vel.x * vel.x + vel.y * vel.y);
    if (speed < MIN_SPEED_FOR_PREDICTION) return path;

    // Check velocity consistency
    float consistency = computeVelocityConsistency();
    if (consistency < MIN_CONSISTENCY) return path;

    // Predict future positions with velocity decay
    float decay = VELOCITY_DECAY;
    for (int i = 1; i <= numFrames; i++) {
        pos = pos + vel;
        vel = vel * decay;
        path.push_back(pos);
    }

    return path;
}

// ============================================================================
// Confidence
// ============================================================================

/// Compute how consistent velocity has been (0-1 based on variance)
float KalmanPredictor::computeVelocityConsistency() const {
    if (velocityHistory.size() < MIN_HISTORY_FOR_MEDIAN) return 0.0f;

    // Compute variance of velocity
    cv::Point2f mean(0, 0);
    for (const auto& v : velocityHistory) {
        mean += v;
    }
    mean = mean * (1.0f / velocityHistory.size());

    float variance = 0.0f;
    for (const auto& v : velocityHistory) {
        cv::Point2f diff = v - mean;
        variance += diff.x * diff.x + diff.y * diff.y;
    }
    variance /= velocityHistory.size();

    // Also check if velocity direction is consistent
    float meanSpeed = std::sqrt(mean.x * mean.x + mean.y * mean.y);
    constexpr float MIN_MEAN_SPEED = 0.5f;
    if (meanSpeed < MIN_MEAN_SPEED) return 0.0f;

    // Lower variance relative to speed = higher consistency
    constexpr float VARIANCE_SCALE = 2.0f;
    float relativeVariance = variance / (meanSpeed * meanSpeed + 1.0f);
    return 1.0f / (1.0f + relativeVariance * VARIANCE_SCALE);
}

/// Get prediction confidence based on consistency, update count, and speed
float KalmanPredictor::getConfidence() const {
    if (!initialized || updateCount < MIN_UPDATES_FOR_CONFIDENCE) return 0.0f;

    float speed = std::sqrt(stableVelocity.x * stableVelocity.x +
                            stableVelocity.y * stableVelocity.y);

    if (speed < MIN_SPEED_FOR_CONFIDENCE) return 0.0f;

    float consistency = computeVelocityConsistency();
    constexpr float COUNT_RAMP_FRAMES = 20.0f;
    float countFactor = std::min(1.0f, (updateCount - MIN_UPDATES_FOR_PREDICTION) / COUNT_RAMP_FRAMES);
    float speedFactor = std::min(1.0f, speed / SPEED_FACTOR_SCALE);

    return consistency * countFactor * (CONFIDENCE_BASE + CONFIDENCE_SPEED_WEIGHT * speedFactor);
}
