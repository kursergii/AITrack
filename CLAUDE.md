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
./bin/aitrack [video_path] [yolo_model] [backbone_model] [neckhead_model]
```

Defaults:
- Video: `../videos/car1.mp4`
- YOLO: `../models/yolo11n.onnx`
- Backbone: `../models/nanotrack_backbone_sim.onnx`
- Neckhead: `../models/nanotrack_head_sim.onnx`

Use `0` for webcam. YOLO model is optional - detection features are disabled if not found.

## Controls

- `q` / `ESC` - Quit
- `a` / `r` - Add new track (select ROI)
- `c` - Clear all tracks
- `m` - Toggle trajectory map
- `+` / `-` - Adjust target FPS
- `SPACE` - Pause/Resume
- `d` - Toggle YOLO detection (if model loaded)
- `t` - Toggle auto-track detections
- `[` / `]` - Decrease/increase detection confidence threshold

## Architecture

```
main.cpp              Entry point
src/
  application.cpp     Main loop, UI, video I/O, trajectory map, detection integration
  nanotracker.cpp     NanoTracker wrapper - legacy, not used
  tracker_manager.cpp Multi-object tracking with trajectory history
  detector.cpp        YOLOv5/v8/v11 ONNX detection with letterbox preprocessing
  kalman_predictor.cpp Motion prediction (constant velocity model)
  camera_motion.cpp   Global camera motion estimation (optical flow)
  object_motion.cpp   Per-object motion relative to camera
  visualization.cpp   Drawing helpers
include/
  *.hpp               Headers for above
```

## Key Components

- **TrackerManager**: Multi-object tracking with per-object NanoTracker, trajectory history (300 frames)
- **KalmanPredictor**: Predicts future trajectory (30 frames) with confidence scoring
- **CameraMotion**: Estimates global motion from optical flow (goodFeaturesToTrack + LK)
- **ObjectMotion**: Computes object velocity relative to environment
- **Detector**: YOLOv5/v8/v11 object detection with letterbox preprocessing, NMS, 80 COCO classes

## Dependencies

- OpenCV (core, highgui, imgproc, video, videoio, tracking, dnn)
- ONNX models in `models/`:
  - NanoTracker: `nanotrack_backbone_sim.onnx`, `nanotrack_head_sim.onnx`
  - YOLO (optional): `yolo11n.onnx` or any YOLOv5/v8/v11 ONNX model

## Session Notes

### 2026-01-27: YOLOv11 with Letterbox Preprocessing

Upgraded detector with proper letterbox preprocessing (from Qt/onnx_opencv reference):
- **Letterbox resize**: Maintains aspect ratio, pads with gray (114) to 640x640
- **Coordinate conversion**: Properly removes padding and scales back to original image
- **YOLOv11 support**: Same output format as v8 [1, 84, 8400], works out of box
- **Model**: Using `yolo11n.onnx` from Qt project (latest Ultralytics model)

### 2026-01-27: YOLO Detection Integration

Added YOLOv5/v8/v11 object detection support:
- **Detector class**: Loads ONNX models, auto-detects YOLOv5 vs v8/v11 format
- **Detection modes**: Manual (shows corner brackets) or auto-track (creates tracks automatically)
- **Integration**: Detection runs every 10 frames when enabled
- **TrackerManager.update()**: Matches detections to existing tracks via IoU, creates new tracks for unmatched
- Controls: `d` toggle detection, `t` toggle auto-track, `[`/`]` adjust threshold (0.1-0.9)

### 2025-01-27: Trajectory Map Added

- Added `trajectory` deque to TrackedObject (max 300 positions)
- Added `renderTrajectoryMap()` - mini-map overlay showing movement history
- Map shows fading trails with current position markers
- Toggle with 'm' key

### 2025-01-27: TrackerManager Integration

Refactored Application to use TrackerManager for multi-object tracking:
- `selectROI()`: reads fresh frame, updates existing trackers, shows them during selection
- Each TrackedObject contains NanoTracker, KalmanPredictor, ObjectMotion, and trajectory
- NanoTracker class is now unused (TrackerManager creates cv::TrackerNano directly)

### Performance Note

FPS counter now includes camera motion computation (optical flow), which is expensive.
The old code only measured tracker update time. Actual tracking performance is similar.
