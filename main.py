import os
import sys
import argparse
import cv2
import torch
import numpy as np
from pathlib import Path
import requests
import zipfile
import tempfile
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from v12.model.yolo import YOLOv12FallDetection

# Fall detection datasets URLs
FALL_DATASETS = {
    "ur_fall": "http://fenix.ur.edu.pl/~mkepski/ds/data/fall-detection-dataset.zip",
    "multicam": "http://fenix.ur.edu.pl/~mkepski/ds/data/multicam-fall-dataset.zip"
}

def download_dataset(dataset_name, output_dir="datasets"):
    """
    Download and extract a fall detection dataset
    
    Args:
        dataset_name: Name of the dataset to download (ur_fall or multicam)
        output_dir: Directory to save the dataset
    """
    if dataset_name not in FALL_DATASETS:
        print(f"Dataset {dataset_name} not found. Available datasets: {list(FALL_DATASETS.keys())}")
        return False
    
    url = FALL_DATASETS[dataset_name]
    output_path = Path(output_dir) / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check if dataset already exists
    if any(output_path.iterdir()):
        print(f"Dataset {dataset_name} already exists at {output_path}")
        return True
    
    print(f"Downloading {dataset_name} dataset from {url}")
    
    # Download the dataset
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_file:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=dataset_name) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_file.write(chunk)
                    pbar.update(len(chunk))
        
        temp_path = temp_file.name
    
    # Extract the dataset
    print(f"Extracting dataset to {output_path}")
    with zipfile.ZipFile(temp_path, 'r') as zip_ref:
        zip_ref.extractall(output_path)
    
    # Clean up
    os.unlink(temp_path)
    print(f"Dataset {dataset_name} downloaded and extracted to {output_path}")
    return True

def test_on_image(model_path=None, image_path=None):
    """
    Test fall detection on a single image
    
    Args:
        model_path: Path to the YOLO model
        image_path: Path to the image to test
    """
    # Initialize model
    model = YOLOv12FallDetection(model_path)
    
    # Use webcam if no image path is provided
    if image_path is None:
        print("No image path provided, using webcam")
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image from webcam")
            return
        cap.release()
    else:
        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Failed to read image from {image_path}")
            return
    
    # Perform fall detection
    annotated_img, fall_detected, fall_confidence = model.detect_fall(frame)
    
    # Display results
    print(f"Fall detected: {fall_detected}, Confidence: {fall_confidence:.2f}")
    
    # Save and show the result
    result_path = "fall_detection_result.jpg"
    cv2.imwrite(result_path, annotated_img)
    print(f"Result saved to {result_path}")
    
    # Display the image
    cv2.imshow("Fall Detection Result", annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_on_video(model_path=None, video_path=None, output_path="fall_detection_output.mp4"):
    """
    Test fall detection on a video
    
    Args:
        model_path: Path to the YOLO model
        video_path: Path to the video to test
        output_path: Path to save the output video
    """
    # Initialize model
    model = YOLOv12FallDetection(model_path)
    
    # Use webcam if no video path is provided
    if video_path is None:
        print("No video path provided, using webcam")
        video_path = 0  # Use default camera
    
    # Process the video
    fall_events = model.process_video(video_path, output_path)
    
    # Display results
    print(f"Detected {len(fall_events)} fall events:")
    for i, event in enumerate(fall_events):
        start_time = event['start_time']
        end_time = event['end_time']
        confidence = event['confidence']
        duration = end_time - start_time if end_time else "ongoing"
        
        print(f"Fall {i+1}: Start: {start_time:.2f}s, End: {end_time:.2f}s, "
              f"Duration: {duration}, Confidence: {confidence:.2f}")
    
    print(f"Output video saved to {output_path}")

def start_api_server(host="0.0.0.0", port=8000):
    """
    Start the Unicorn API server
    
    Args:
        host: Host to bind the server to
        port: Port to bind the server to
    """
    print(f"Starting Fall Detection API server at http://{host}:{port}")
    import uvicorn
    import importlib.util
    
    # Import the app directly from the server module
    server_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v12/api/server.py")
    spec = importlib.util.spec_from_file_location("server_module", server_path)
    server_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server_module)
    
    # Run the app using uvicorn
    uvicorn.run(server_module.app, host=host, port=port)

def main():
    """Main function to parse arguments and run the appropriate command"""
    parser = argparse.ArgumentParser(description="YOLOv12 Fall Detection")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Download dataset command
    download_parser = subparsers.add_parser("download", help="Download fall detection dataset")
    download_parser.add_argument("dataset", choices=list(FALL_DATASETS.keys()), 
                                help="Dataset to download")
    download_parser.add_argument("--output", default="datasets", 
                                help="Directory to save the dataset")
    
    # Test on image command
    image_parser = subparsers.add_parser("image", help="Test fall detection on an image")
    image_parser.add_argument("--model", help="Path to YOLO model")
    image_parser.add_argument("--image", help="Path to image file")
    
    # Test on video command
    video_parser = subparsers.add_parser("video", help="Test fall detection on a video")
    video_parser.add_argument("--model", help="Path to YOLO model")
    video_parser.add_argument("--video", help="Path to video file")
    video_parser.add_argument("--output", default="fall_detection_output.mp4", 
                             help="Path to save output video")
    
    # Start API server command
    server_parser = subparsers.add_parser("server", help="Start the API server")
    server_parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    server_parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    
    args = parser.parse_args()
    
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Run the appropriate command
    if args.command == "download":
        download_dataset(args.dataset, args.output)
    elif args.command == "image":
        test_on_image(args.model, args.image)
    elif args.command == "video":
        test_on_video(args.model, args.video, args.output)
    elif args.command == "server":
        start_api_server(args.host, args.port)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
