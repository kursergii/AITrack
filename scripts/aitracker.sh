#!/bin/bash
# AITracker wrapper - reads config and runs tracker
# Usage: ./scripts/aitracker.sh video.mp4 [yolo_model]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$PROJECT_DIR/config/aitracker.yaml"

# Default model paths
YOLO_MODEL="$PROJECT_DIR/models/drones.onnx"
BACKBONE="$PROJECT_DIR/models/nanotrack_backbone_sim.onnx"
NECKHEAD="$PROJECT_DIR/models/nanotrack_head_sim.onnx"

# Parse config if exists
if [ -f "$CONFIG_FILE" ]; then
    # Extract model paths from YAML (simple grep)
    YOLO_CFG=$(grep "yolo:" "$CONFIG_FILE" | sed 's/.*yolo: *//' | tr -d '"')
    BACKBONE_CFG=$(grep "nanotrack_backbone:" "$CONFIG_FILE" | sed 's/.*nanotrack_backbone: *//' | tr -d '"')
    NECKHEAD_CFG=$(grep "nanotrack_head:" "$CONFIG_FILE" | sed 's/.*nanotrack_head: *//' | tr -d '"')

    # Use config paths if they exist
    [ -n "$YOLO_CFG" ] && YOLO_MODEL="$PROJECT_DIR/$YOLO_CFG"
    [ -n "$BACKBONE_CFG" ] && BACKBONE="$PROJECT_DIR/$BACKBONE_CFG"
    [ -n "$NECKHEAD_CFG" ] && NECKHEAD="$PROJECT_DIR/$NECKHEAD_CFG"
fi

# Override YOLO model if provided as argument
if [ -n "$2" ]; then
    YOLO_MODEL="$2"
fi

# Check video argument
if [ -z "$1" ]; then
    echo "Usage: $0 <video_source> [yolo_model]"
    echo ""
    echo "Examples:"
    echo "  $0 video.mp4"
    echo "  $0 0                    # webcam"
    echo "  $0 video.mp4 custom.onnx"
    exit 1
fi

VIDEO="$1"

# Check files exist
if [ ! -f "$YOLO_MODEL" ]; then
    echo "Error: YOLO model not found: $YOLO_MODEL"
    exit 1
fi

if [ ! -f "$BACKBONE" ]; then
    echo "Error: NanoTrack backbone not found: $BACKBONE"
    exit 1
fi

if [ ! -f "$NECKHEAD" ]; then
    echo "Error: NanoTrack head not found: $NECKHEAD"
    exit 1
fi

echo "AITracker"
echo "  Video:    $VIDEO"
echo "  YOLO:     $YOLO_MODEL"
echo "  Backbone: $BACKBONE"
echo "  Head:     $NECKHEAD"
echo ""

# Run aitracker
exec "$PROJECT_DIR/bin/aitracker" "$VIDEO" "$YOLO_MODEL" "$BACKBONE" "$NECKHEAD"
