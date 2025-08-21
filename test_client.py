#!/usr/bin/env python
"""
Test client for the Fall Detection API
"""
import os
import sys
import requests
import argparse
from pathlib import Path
import cv2
import numpy as np
import time
import json

def test_api_status(base_url="http://localhost:8000"):
    """Test if the API is running"""
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print(f"✅ API is running: {response.json()}")
            return True
        else:
            print(f"❌ API returned status code {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the API server")
        return False

def test_image_detection(image_path, base_url="http://localhost:8000", model="YOLOv8"):
    """Test fall detection on an image"""
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return
    
    # Prepare the file for upload
    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
        data = {"model": model}
        
        # Send the request
        print(f"Sending image {image_path} to API using {model}...")
        try:
            response = requests.post(f"{base_url}/detect", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Detection successful:")
                print(f"  - Fall detected: {result['fall_detected']}")
                print(f"  - Confidence: {result['confidence']:.2f}")
                print(f"  - Model: {result.get('model_version', model)}")
                print(f"  - Result image: {result['result_image']}")
                return result
            else:
                print(f"❌ API returned status code {response.status_code}")
                print(response.text)
                return None
        except requests.exceptions.ConnectionError:
            print("❌ Could not connect to the API server")
            return None

def test_video_detection(video_path, base_url="http://localhost:8000", model="YOLOv8"):
    """Test fall detection on a video"""
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return
    
    # Prepare the file for upload
    with open(video_path, "rb") as f:
        files = {"file": (os.path.basename(video_path), f, "video/mp4")}
        data = {"model": model}
        
        # Send the request
        print(f"Sending video {video_path} to API for processing using {model}...")
        try:
            response = requests.post(f"{base_url}/process-video", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                job_id = result["job_id"]
                print(f"✅ Video processing started:")
                print(f"  - Job ID: {job_id}")
                print(f"  - Status: {result['status']}")
                print(f"  - Model: {model}")
                
                # Poll for job completion
                print("Polling for job completion...")
                while True:
                    job_status = check_job_status(job_id, base_url)
                    if job_status["status"] == "completed":
                        print("✅ Video processing completed!")
                        print(f"  - Fall events: {len(job_status['results']['fall_events'])}")
                        print(f"  - Model: {job_status['results'].get('model_version', model)}")
                        print(f"  - Output video: {job_status['results']['output_video']}")
                        return job_status
                    elif job_status["status"] == "error":
                        print(f"❌ Video processing failed: {job_status['error']}")
                        return job_status
                    
                    print(f"  - Status: {job_status['status']}, waiting...")
                    time.sleep(5)
            else:
                print(f"❌ API returned status code {response.status_code}")
                print(response.text)
                return None
        except requests.exceptions.ConnectionError:
            print("❌ Could not connect to the API server")
            return None

def check_job_status(job_id, base_url="http://localhost:8000"):
    """Check the status of a video processing job"""
    try:
        response = requests.get(f"{base_url}/job/{job_id}")
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "error", "error": f"API returned status code {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"status": "error", "error": "Could not connect to the API server"}

def capture_from_webcam(output_path="webcam_capture.jpg"):
    """Capture an image from the webcam"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return None
    
    # Capture a frame
    ret, frame = cap.read()
    if not ret:
        print("❌ Could not capture frame from webcam")
        cap.release()
        return None
    
    # Save the frame
    cv2.imwrite(output_path, frame)
    print(f"✅ Captured image from webcam and saved to {output_path}")
    
    # Release the webcam
    cap.release()
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Test client for Fall Detection API")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check if the API is running")
    status_parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the API")
    
    # Image command
    image_parser = subparsers.add_parser("image", help="Test fall detection on an image")
    image_parser.add_argument("--image", help="Path to image file")
    image_parser.add_argument("--webcam", action="store_true", help="Capture image from webcam")
    image_parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the API")
    image_parser.add_argument("--model", default="YOLOv8", choices=["YOLOv8", "YOLOv11"], help="YOLO model to use")
    
    # Video command
    video_parser = subparsers.add_parser("video", help="Test fall detection on a video")
    video_parser.add_argument("--video", required=True, help="Path to video file")
    video_parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the API")
    video_parser.add_argument("--model", default="YOLOv8", choices=["YOLOv8", "YOLOv11"], help="YOLO model to use")
    
    args = parser.parse_args()
    
    if args.command == "status":
        test_api_status(args.url)
    elif args.command == "image":
        image_path = None
        if args.webcam:
            image_path = capture_from_webcam()
        elif args.image:
            image_path = args.image
        else:
            print("❌ Either --image or --webcam must be specified")
            return
        
        if image_path:
            test_image_detection(image_path, args.url, args.model)
    elif args.command == "video":
        test_video_detection(args.video, args.url, args.model)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
