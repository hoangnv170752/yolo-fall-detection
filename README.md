# YOLO Fall Detection

A system for detecting falls using YOLOv12 with PyTorch and Unicorn server. This project provides tools to detect falls in images and videos, as well as an API server for real-time fall detection.

## Features

- Fall detection in images and videos using YOLOv12
- API server using Unicorn (uvicorn) and FastAPI
- Web interface for drag-and-drop file testing and benchmarking
- Support for downloading fall detection datasets
- Real-time fall detection using webcam
- Batch processing of videos

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/yolo-fall-detection.git
cd yolo-fall-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Download Fall Detection Dataset

```bash
python main.py download ur_fall --output datasets
```

Available datasets:
- `ur_fall`: UR Fall Detection dataset
- `multicam`: Multicam Fall dataset

### Test on Image

```bash
python main.py image --image path/to/image.jpg
```

If no image is provided, it will use your webcam to capture an image.

### Test on Video

```bash
python main.py video --video path/to/video.mp4 --output results.mp4
```

If no video is provided, it will use your webcam for real-time fall detection.

### Start API Server (Unicorn)

Use the standalone server script for the most reliable way to start the server:

```bash
python run_server.py
```

Alternatively, you can use the main.py command (though this might have issues in some environments):

```bash
python main.py server --host 0.0.0.0 --port 8000
```

This will start the Unicorn server at http://localhost:8000

### Web Interface

After starting the server, access the web interface by opening http://localhost:8000 in your browser. The web interface provides:

- Drag-and-drop functionality for uploading images and videos
- Visual results of fall detection
- Benchmarking statistics including processing time and confidence scores
- Detailed results table for all processed files

## API Endpoints

- `GET /`: Web interface for fall detection
- `GET /api/status`: Check if the API is running
- `POST /detect`: Detect falls in an uploaded image
- `POST /process-video`: Process a video file for fall detection
- `GET /job/{job_id}`: Get the status of a video processing job

## Project Structure

```
├── main.py                 # Main entry point
├── requirements.txt        # Dependencies
├── v12/
│   ├── api/
│   │   └── server.py       # Unicorn API server
│   └── model/
│       └── yolo.py         # YOLOv12 model implementation
└── datasets/               # Downloaded datasets (created when needed)
```

## How It Works

The system uses YOLOv12 to detect people in frames and then analyzes their posture to determine if they have fallen. The detection is based on the aspect ratio of the bounding box around a person - when a person falls, their bounding box typically becomes wider than tall.

For more accurate detection, the system can also analyze temporal information across multiple frames to detect sudden changes in position that might indicate a fall.
