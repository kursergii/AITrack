# AITrack - Multi-Object Tracking System

Real-time multi-object tracking combining YOLO detection with NanoTracker visual tracking, featuring motion prediction, camera motion compensation, and trajectory visualization.

## Features

- **Multi-Object Tracking**: Track multiple objects simultaneously with unique IDs
- **YOLO Detection**: Automatic object detection using YOLOv5/v8/v11 ONNX models
- **NanoTracker**: Fast siamese-network based visual tracking for each object
- **Motion Prediction**: Kalman filter-based trajectory forecasting with visualization
- **Camera Motion Compensation**: Optical flow-based camera motion estimation
- **Object Motion Analysis**: True object motion in world coordinates (camera-compensated)
- **Async Detection**: Non-blocking detection on separate thread for smooth playback
- **GPU Acceleration**: Automatic CUDA/OpenCL backend selection
- **YAML Configuration**: Runtime-configurable settings without recompiling
- **Class Filtering**: Allow/block specific object classes for targeted tracking

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

The executable will be in `bin/aitracker`.

## Models

Place the following ONNX models in the `models/` folder:

| Model | Description |
|-------|-------------|
| `nanotrack_backbone_sim.onnx` | NanoTracker backbone network |
| `nanotrack_head_sim.onnx` | NanoTracker head network |
| `yolo11n.onnx` | YOLO detector (general objects) |
| `drones.onnx` | YOLO detector (drone-specific, optional) |

NanoTracker models are required for tracking. YOLO model is optional - without it, you can manually select objects to track.

## Usage

### Using the wrapper script (recommended)

The wrapper script reads config and resolves model paths automatically:

```bash
# Basic usage
./scripts/aitracker.sh path/to/video.mp4

# With custom YOLO model
./scripts/aitracker.sh video.mp4 path/to/custom_yolo.onnx

# Webcam
./scripts/aitracker.sh 0
```

### Direct binary usage

```bash
# Basic usage with video file
./bin/aitracker path/to/video.mp4

# Use webcam
./bin/aitracker 0

# With custom YOLO model
./bin/aitracker video.mp4 path/to/yolo.onnx

# With all custom models
./bin/aitracker video.mp4 yolo.onnx backbone.onnx neckhead.onnx
```

**Arguments:**
- `video_source` (required): Path to video file or `0` for webcam
- `yolo_model` (optional): Path to YOLO ONNX model
- `backbone` (optional): Path to NanoTracker backbone
- `neckhead` (optional): Path to NanoTracker head

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

## Configuration

Settings can be configured via YAML config file at `config/aitracker.yaml`. The application searches for config in these locations (in order):

1. `config/aitracker.yaml`
2. `../config/aitracker.yaml`
3. `aitracker.yaml`

Command-line arguments override config file values.

**Note:** OpenCV's FileStorage requires YAML files to start with `%YAML:1.0` as the first line.

Example config:

```yaml
%YAML:1.0

tracker:
  max_lost_frames: 10       # Frames before track is removed
  min_iou: 0.5              # Detection-track match threshold
  reinit_threshold: 0.7     # IoU threshold for tracker reinitialization
  max_trajectory_length: 30 # Max positions in trajectory history

detection:
  confidence: 0.5            # YOLO confidence threshold
  nms_threshold: 0.45        # Non-maximum suppression threshold
  interval: 10               # Run detection every N frames
  input_size: 640            # YOLO input size

display:
  target_fps: 60
  show_trajectory_map: false
  auto_track_detections: false

models:
  yolo: models/drones.onnx
  nanotrack_backbone: models/nanotrack_backbone_sim.onnx
  nanotrack_head: models/nanotrack_head_sim.onnx

classes:
  allowed: ["drone"]         # Only track these classes (empty = all)
  blocked: []                # Never track these classes
```

## Architecture

```
Application
    ├── VideoCapture (frame source)
    ├── Config (YAML file + CLI overrides)
    ├── TrackerManager (multi-object tracking)
    │   ├── TrackedObject[]
    │   │   ├── NanoTracker (visual tracker)
    │   │   ├── KalmanPredictor (motion prediction)
    │   │   ├── ObjectMotion (camera-compensated motion)
    │   │   └── trajectory (position history)
    │   └── Detection-Track matching (IoU-based)
    ├── Detector (YOLO, async thread)
    │   ├── Preprocess (letterbox)
    │   ├── Forward (CUDA/OpenCL/CPU)
    │   └── Postprocess (NMS)
    ├── CameraMotion (sparse optical flow)
    └── Visualization
        ├── Predicted paths
        ├── Motion indicators (camera + object)
        └── Trajectory map
```

## Pipeline

1. **Frame Capture**: Read frame from video/camera
2. **Camera Motion**: Estimate global motion using sparse optical flow (Lucas-Kanade)
3. **Detection** (async): Run YOLO on separate thread every N frames
4. **Tracking**:
   - Run NanoTracker on all active tracks
   - Match new detections to existing tracks (greedy IoU matching)
   - Update matched tracks with detection positions
   - Create new tracks for unmatched detections (respecting class filters)
   - Remove tracks lost for too long
5. **Motion Analysis**: Compute object motion with camera compensation
6. **Prediction**: Kalman filter predicts future positions with velocity smoothing
7. **Render**: Draw bounding boxes, predictions, motion indicators, and trajectory map

## File Structure

```
AITrack/
├── main.cpp                 # Entry point
├── CMakeLists.txt           # Build configuration
├── include/
│   ├── application.hpp      # Main application controller
│   ├── config.hpp           # Configuration with YAML loading
│   ├── tracker_manager.hpp  # Multi-object tracker
│   ├── detector.hpp         # YOLO detector
│   ├── kalman_predictor.hpp # Motion prediction
│   ├── camera_motion.hpp    # Optical flow camera motion
│   ├── object_motion.hpp    # Object motion compensation
│   ├── visualization.hpp    # Drawing utilities
│   └── nanotracker.hpp      # NanoTracker wrapper
├── src/                     # Implementation files
├── config/
│   └── aitracker.yaml       # Runtime configuration
├── scripts/
│   └── aitracker.sh         # Wrapper script
├── models/                  # ONNX model files
└── videos/                  # Test videos
```

## License

MIT License
