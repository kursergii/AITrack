# AITrack

C++17 video object tracking application using OpenCV and NanoTracker.

## Build

```bash
cd build
cmake ..
make
```

Executable outputs to `bin/aitrack`.

## Run

```bash
./bin/aitrack [video_path] [backbone_model] [neckhead_model]
```

Defaults:
- Video: `../videos/car1.mp4`
- Backbone: `../models/nanotrack_backbone_sim.onnx`
- Neckhead: `../models/nanotrack_head_sim.onnx`

Use `0` for webcam.

## Controls

- `q` / `ESC` - Quit
- `a` / `r` - Add new track (select ROI)
- `c` - Clear all tracks
- `+` / `-` - Adjust target FPS
- `SPACE` - Pause/Resume

## Architecture

```
main.cpp              Entry point
src/
  application.cpp     Main loop, UI, video I/O (uses TrackerManager)
  nanotracker.cpp     NanoTracker wrapper (cv::TrackerNano) - legacy, not used
  tracker_manager.cpp Multi-object tracking with detection matching
  detector.cpp        Object detection interface (for future use)
  kalman_predictor.cpp Motion prediction (constant velocity model)
  camera_motion.cpp   Global camera motion estimation
  object_motion.cpp   Per-object motion relative to camera
  visualization.cpp   Drawing helpers
include/
  *.hpp               Headers for above
```

## Key Components

- **TrackerManager**: Multi-object tracking - creates NanoTracker per object, handles IoU-based detection matching
- **KalmanPredictor**: Predicts future trajectory (30 frames) with confidence scoring
- **CameraMotion**: Estimates global motion from optical flow
- **ObjectMotion**: Computes object velocity relative to environment

## Dependencies

- OpenCV (core, highgui, imgproc, video, videoio, tracking)
- ONNX models in `models/`

## Session Notes

### 2025-01-27: TrackerManager Integration Complete

Refactored Application to use TrackerManager for multi-object tracking:
- `init()`: calls `trackerManager.setModelPaths()` instead of loading single tracker
- `selectROI()`: calls `trackerManager.addTrack()` to add objects
- `processFrame()`: calls `trackerManager.updateTrackers()` which handles motion/prediction per object
- `render()`: iterates over `trackerManager.getTrackedObjects()`, draws each with unique color
- `handleInput()`: added 'a' (add track) and 'c' (clear all) controls

Each TrackedObject contains its own NanoTracker, KalmanPredictor, and ObjectMotion.

NanoTracker class is now unused (TrackerManager creates cv::TrackerNano directly).
