# AITrack - Multi-Object Tracking System

Real-time multi-object tracking combining YOLO detection with NanoTracker visual tracking, featuring motion prediction and trajectory visualization.

## Features

- **Multi-Object Tracking**: Track multiple objects simultaneously with unique IDs
- **YOLO Detection**: Automatic object detection using YOLOv5/v8/v11 ONNX models
- **NanoTracker**: Fast siamese-network based visual tracking for each object
- **Motion Prediction**: Kalman filter-based trajectory forecasting with visualization
- **Camera Motion Compensation**: Optical flow-based camera motion estimation
- **Async Detection**: Non-blocking detection on separate thread for smooth playback
- **GPU Acceleration**: Automatic CUDA/OpenCL backend selection

## Requirements

- CMake 3.10+
- C++17 compiler
- OpenCV 4.x with:
  - core, highgui, imgproc, video, videoio
  - tracking (for NanoTracker)
  - dnn (for YOLO detection)
  - cuda (optional, for GPU acceleration)

## Building

```bash
mkdir build && cd build
cmake ..
make
```

The executable will be in `bin/aitrack`.

## Models

Place the following ONNX models in the `models/` folder:

| Model | Description |
|-------|-------------|
| `nanotrack_backbone_sim.onnx` | NanoTracker backbone network |
| `nanotrack_head_sim.onnx` | NanoTracker head network |
| `yolo11n.onnx` | YOLO detector (optional) |

NanoTracker models are required for tracking. YOLO model is optional - without it, you can manually select objects to track.

## Usage

```bash
# Basic usage with video file
./bin/aitrack path/to/video.mp4

# Use webcam (device index: 0, 1, 2, ...)
./bin/aitrack 0    # First camera
./bin/aitrack 1    # Second camera

# With custom YOLO model
./bin/aitrack video.mp4 path/to/yolo.onnx

# With all custom models
./bin/aitrack video.mp4 yolo.onnx backbone.onnx neckhead.onnx
```

**Arguments:**
- `video_source` (required): Path to video file or camera index (`0`, `1`, `2`, ...)
- `yolo_model` (optional): Path to YOLO ONNX model (default: `../models/yolo11n.onnx`)
- `backbone` (optional): Path to NanoTracker backbone (default: `../models/nanotrack_backbone_sim.onnx`)
- `neckhead` (optional): Path to NanoTracker head (default: `../models/nanotrack_head_sim.onnx`)

## Controls

| Key | Action |
|-----|--------|
| `q` / `ESC` | Quit |
| `a` / `r` | Add new track (ROI selection) |
| `c` | Clear all tracks |
| `SPACE` | Pause/Resume |
| `d` | Toggle detection |
| `t` | Toggle auto-tracking of detections |
| `m` | Toggle trajectory map |
| `+` / `-` | Adjust target FPS |
| `[` / `]` | Adjust detection threshold |

## Architecture

```
Application
    ├── VideoCapture (frame source)
    ├── TrackerManager (multi-object tracking)
    │   ├── TrackedObject[]
    │   │   ├── NanoTracker (visual tracker)
    │   │   ├── KalmanPredictor (motion prediction)
    │   │   └── trajectory (position history)
    │   └── Detection-Track matching (IoU-based)
    ├── Detector (YOLO, async thread)
    │   ├── Preprocess (letterbox)
    │   └── Postprocess (NMS)
    └── Visualization
        ├── Predicted paths
        ├── Motion indicators
        └── Trajectory map
```

## Pipeline

1. **Frame Capture**: Read frame from video/camera
2. **Camera Motion**: Estimate global motion using optical flow
3. **Detection** (async): Run YOLO on separate thread every N frames
4. **Tracking**:
   - Run NanoTracker on all active tracks
   - Match new detections to existing tracks (IoU)
   - Update matched tracks with detection positions
   - Create new tracks for unmatched detections
   - Remove tracks lost for too long
5. **Prediction**: Kalman filter predicts future positions
6. **Render**: Draw bounding boxes, predictions, and overlays

## Configuration

Key parameters in `include/config.hpp`:

```cpp
struct Config {
    int maxLostFrames = 30;        // Frames before track removed
    float minIoU = 0.3f;           // Detection-track match threshold
    float detectionConfidence = 0.5f;
    int detectionInterval = 10;    // Run detection every N frames
    int targetFps = 30;
};
```

## File Structure

```
aitrack/
├── main.cpp                 # Entry point
├── include/
│   ├── application.hpp      # Main application controller
│   ├── tracker_manager.hpp  # Multi-object tracker
│   ├── detector.hpp         # YOLO detector
│   ├── kalman_predictor.hpp # Motion prediction
│   ├── camera_motion.hpp    # Optical flow camera motion
│   ├── object_motion.hpp    # Object motion compensation
│   ├── visualization.hpp    # Drawing utilities
│   ├── nanotracker.hpp      # NanoTracker wrapper
│   └── config.hpp           # Configuration
├── src/
│   ├── application.cpp
│   ├── tracker_manager.cpp
│   ├── detector.cpp
│   ├── kalman_predictor.cpp
│   ├── camera_motion.cpp
│   ├── object_motion.cpp
│   ├── visualization.cpp
│   └── nanotracker.cpp
├── models/                  # ONNX model files
├── videos/                  # Test videos
└── CMakeLists.txt
```

## License

MIT License
