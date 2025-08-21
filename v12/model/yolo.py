import torch
import torch.nn as nn
from ultralytics import YOLO
import os
import numpy as np
import cv2
import subprocess
import sys
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

class YOLOFallDetection:
    """
    YOLO model for fall detection using the Ultralytics YOLO implementation.
    This class provides methods to load the model, perform inference, and process results.
    Supports YOLOv8 and YOLOv8-P2 models.
    """
    
    def __init__(self, model_version: str = "v8", model_path: Optional[str] = None, conf_threshold: float = 0.25):
        """
        Initialize the YOLO model for fall detection.
        
        Args:
            model_version: YOLO version to use ("v8" or "v8p2")
            model_path: Path to a pre-trained YOLO model. If None, will use the default model for the specified version.
            conf_threshold: Confidence threshold for detections.
        """
        self.conf_threshold = conf_threshold
        self.model_version = model_version.lower()
        
        # Load the appropriate model based on version
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
            print(f"Loaded custom model from {model_path}")
        else:
            try:
                # Use default models based on version
                if self.model_version == "v8p2":
                    # Try to load or install YOLOv8-P2
                    v8p2_path = self._get_yolov8p2_model()
                    if v8p2_path:
                        self.model = YOLO(v8p2_path)
                        print(f"Loaded YOLOv8-P2 model from {v8p2_path}")
                    else:
                        print("Could not load YOLOv8-P2 model. Using YOLOv8n as fallback.")
                        self.model = YOLO('yolov8n.pt')  # This will auto-download if needed
                        print("Using YOLOv8n model as fallback for YOLOv8-P2")
                else:
                    # Default to YOLOv8 model
                    self.model = YOLO('yolov8n.pt')  # This will auto-download if needed
                    print("Using default YOLOv8n model. For fall detection, fine-tuning is recommended.")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Falling back to YOLOv8n model")
                self.model = YOLO('yolov8n.pt')  # This will auto-download if needed
        
        # Classes of interest for fall detection (person is class 0 in COCO)
        self.target_classes = [0]  # Person class
        
        # Fall detection parameters
        self.fall_threshold = 0.6  # Threshold for classifying a fall
        self.history_size = 10  # Number of frames to keep for motion analysis
        self.position_history = []  # Store positions for temporal analysis
    
    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Perform object detection on an image.
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV)
            
        Returns:
            Tuple containing:
                - Annotated image with detections
                - List of detection results
        """
        # Run inference
        results = self.model(image, conf=self.conf_threshold)
        
        # Process results
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Only consider person class (0 in COCO dataset)
                if box.cls.cpu().numpy()[0] in self.target_classes:
                    x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                    confidence = box.conf.cpu().numpy()[0]
                    class_id = int(box.cls.cpu().numpy()[0])
                    
                    detection = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(confidence),
                        'class_id': class_id
                    }
                    detections.append(detection)
        
        # Get annotated image
        annotated_img = results[0].plot()
        
        return annotated_img, detections
    
    def detect_fall(self, image: np.ndarray) -> Tuple[np.ndarray, bool, float]:
        """
        Detect if a person has fallen in the image.
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV)
            
        Returns:
            Tuple containing:
                - Annotated image with fall detection results
                - Boolean indicating if a fall was detected
                - Confidence score of the fall detection
        """
        # Detect people in the image
        annotated_img, detections = self.detect(image)
        
        # No people detected
        if not detections:
            self.position_history.append([])  # Add empty frame to history
            if len(self.position_history) > self.history_size:
                self.position_history.pop(0)
            return annotated_img, False, 0.0
        
        # Process detections to identify falls
        fall_detected = False
        fall_confidence = 0.0
        current_positions = []
        
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Calculate aspect ratio of bounding box
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / height if height > 0 else 0
            
            # Store position for temporal analysis
            centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
            current_positions.append({
                'centroid': centroid,
                'bbox': bbox,
                'aspect_ratio': aspect_ratio
            })
            
            # Simple fall detection based on aspect ratio
            # A person lying down typically has width > height
            if aspect_ratio > 1.5:  # Threshold for considering someone as horizontal
                fall_confidence = max(fall_confidence, min(aspect_ratio / 3.0, 0.95))
                
                # Draw fall detection on the image
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated_img, f"FALL DETECTED: {fall_confidence:.2f}", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                fall_detected = True
            else:
                # Normal standing/sitting person
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Update position history
        self.position_history.append(current_positions)
        if len(self.position_history) > self.history_size:
            self.position_history.pop(0)
        
        # Advanced fall detection using temporal information could be implemented here
        # This would analyze the motion patterns across frames
        
        return annotated_img, fall_detected, fall_confidence
    
    def _get_yolov8p2_model(self) -> Optional[str]:
        """
        Get or install YOLOv8-P2 model (a variant of YOLOv8).
        
        Returns:
            Path to the YOLOv8-P2 model file or None if installation failed
        """
        # Check if YOLOv8-P2 model already exists
        model_path = Path('yolov8n-p2.pt')
        if model_path.exists():
            return str(model_path)
            
        # Try to download YOLOv8-P2 model
        try:
            print("Downloading YOLOv8-P2 model...")
            # Download YOLOv8n-p2 model from Ultralytics
            import requests
            url = "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-p2.pt"
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open('yolov8n-p2.pt', 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("YOLOv8-P2 model downloaded successfully")
                return 'yolov8n-p2.pt'
            else:
                print(f"Failed to download YOLOv8-P2 model: HTTP {response.status_code}")
                return None
        except Exception as e:
            print(f"Error downloading YOLOv8-P2 model: {e}")
            return None

    def process_video(self, video_path: str, output_path: Optional[str] = None) -> List[Dict]:
        """
        Process a video file for fall detection.
        
        Args:
            video_path: Path to the input video file
            output_path: Path to save the output video with annotations
            
        Returns:
            List of fall detection events with timestamps
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process video frames
        fall_events = []
        frame_count = 0
        fall_in_progress = False
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame for fall detection
            annotated_frame, fall_detected, fall_confidence = self.detect_fall(frame)
            
            # Record fall events
            timestamp = frame_count / fps
            
            # Track continuous fall events
            if fall_detected and not fall_in_progress:
                fall_in_progress = True
                fall_events.append({
                    'start_time': timestamp,
                    'end_time': None,
                    'confidence': fall_confidence
                })
            elif not fall_detected and fall_in_progress:
                fall_in_progress = False
                if fall_events:
                    fall_events[-1]['end_time'] = timestamp
            
            # Update confidence for ongoing fall
            if fall_in_progress and fall_events:
                fall_events[-1]['confidence'] = max(fall_events[-1]['confidence'], fall_confidence)
            
            # Write frame to output video
            if writer:
                writer.write(annotated_frame)
            
            frame_count += 1
        
        # Close the last fall event if needed
        if fall_in_progress and fall_events and fall_events[-1]['end_time'] is None:
            fall_events[-1]['end_time'] = frame_count / fps
        
        # Release resources
        cap.release()
        if writer:
            writer.release()
        
        return fall_events
