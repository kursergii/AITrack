#include "kalman_predictor.hpp"
#include <algorithm>
#include <cmath>

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
        0.05f, 0,     0,    0,
        0,     0.05f, 0,    0,
        0,     0,     0.5f, 0,
        0,     0,     0,    0.5f);

    // Measurement noise - higher = trust measurements less
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(10.0f));

    // Initial error covariance
    cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1));
}

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
    float alpha = 0.25f;  // Lower alpha = more smoothing
    smoothedVelocity = smoothedVelocity * (1.0f - alpha) + envVelocity * alpha;

    // Compute extra-stable velocity using median of history
    if (velocityHistory.size() >= 5) {
        stableVelocity = computeMedianVelocity();
    } else {
        stableVelocity = smoothedVelocity;
    }

    prevPosition = measuredPos;
    updateCount++;
}

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

std::vector<cv::Point2f> KalmanPredictor::predictPath(int numFrames) const {
    std::vector<cv::Point2f> path;

    if (!initialized || updateCount < 10) return path;

    // Use Kalman position and stable velocity for prediction
    cv::Point2f pos = getPosition();

    // Blend smoothed and stable velocity for best results
    cv::Point2f vel = stableVelocity * 0.7f + smoothedVelocity * 0.3f;

    // Only predict if there's meaningful and consistent velocity
    float speed = std::sqrt(vel.x * vel.x + vel.y * vel.y);
    if (speed < 1.0f) return path;

    // Check velocity consistency
    float consistency = computeVelocityConsistency();
    if (consistency < 0.3f) return path;

    // Predict future positions with velocity decay
    float decay = 0.97f;
    for (int i = 1; i <= numFrames; i++) {
        pos = pos + vel;
        vel = vel * decay;
        path.push_back(pos);
    }

    return path;
}

float KalmanPredictor::computeVelocityConsistency() const {
    if (velocityHistory.size() < 5) return 0.0f;

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
    if (meanSpeed < 0.5f) return 0.0f;

    // Lower variance relative to speed = higher consistency
    float relativeVariance = variance / (meanSpeed * meanSpeed + 1.0f);
    return 1.0f / (1.0f + relativeVariance * 2.0f);
}

float KalmanPredictor::getConfidence() const {
    if (!initialized || updateCount < 12) return 0.0f;

    float speed = std::sqrt(stableVelocity.x * stableVelocity.x +
                            stableVelocity.y * stableVelocity.y);

    if (speed < 1.0f) return 0.0f;

    float consistency = computeVelocityConsistency();
    float countFactor = std::min(1.0f, (updateCount - 10) / 20.0f);
    float speedFactor = std::min(1.0f, speed / 8.0f);

    return consistency * countFactor * (0.4f + 0.6f * speedFactor);
}
