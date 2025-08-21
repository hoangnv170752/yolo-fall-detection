from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, Query, Path
from fastapi.responses import StreamingResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
import os
import tempfile
import shutil
import time
from typing import List, Dict, Optional, Any
import sys
import logging
from datetime import datetime
import json
import io
from .database import FallDetectionDB

# Add project root to path to import model
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v12.model.yolo import YOLOFallDetection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("fall-detection-api")

# Initialize FastAPI app
app = FastAPI(
    title="Fall Detection API",
    description="API for detecting falls using YOLOv12",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
models = {}
db = None
temp_dir = tempfile.mkdtemp()
results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(results_dir, exist_ok=True)

# Create a directory for storing uploaded files
uploads_dir = os.path.join(temp_dir, "uploads")
os.makedirs(uploads_dir, exist_ok=True)

@app.on_event("startup")
async def startup_event():
    """Initialize the models and database when the server starts"""
    global models, db
    
    # Initialize YOLO models for different versions
    logger.info("Initializing YOLO models...")
    try:
        yolov8_model = YOLOFallDetection(model_version="v8")
        logger.info("YOLOv8 model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize YOLOv8 model: {e}")
        yolov8_model = None

    try:
        yolov8p2_model = YOLOFallDetection(model_version="v8p2")
        logger.info("YOLOv8-P2 model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize YOLOv8-P2 model: {e}")
        yolov8p2_model = None  
    logger.info("Initializing database...")
    db = FallDetectionDB()
    logger.info("Database initialized successfully")

    # Make models available to endpoints
    app.state.yolov8_model = yolov8_model
    app.state.yolov8p2_model = yolov8p2_model

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources when the server shuts down"""
    global temp_dir
    logger.info("Cleaning up temporary files...")
    shutil.rmtree(temp_dir, ignore_errors=True)

@app.get("/")
async def root():
    """Root endpoint - redirects to web interface"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/index.html")

@app.get("/api/status")
async def api_status():
    """API status endpoint"""
    return {"message": "Fall Detection API is running", "status": "active"}

@app.post("/detect")
async def detect_falls(file: UploadFile = File(...), model: str = Form("YOLOv8")):
    """
    Detect falls in an image
    
    Args:
        file: Image file to analyze
        model: YOLO model version to use (YOLOv8 or YOLOv8-P2)
    
    Returns:
        JSON response with detection results
    """
    # Validate file
    if not file.content_type.startswith("image/"):
        return JSONResponse(
            status_code=400,
            content={"error": "Only image files are supported"}
        )
    
    # Select the appropriate model based on the request
    if model == "YOLOv8":
        selected_model = app.state.yolov8_model
    elif model == "YOLOv8-P2":
        selected_model = app.state.yolov8p2_model
    else:
        raise Exception(f"Invalid model: {model}. Supported models: YOLOv8, YOLOv8-P2")
    
    logger.info(f"Processing image with {model}")
    start_time = time.time()
    
    # Read and process the image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Could not decode image"}
        )
    
    # Perform fall detection with selected model
    annotated_img, fall_detected, fall_confidence = selected_model.detect_fall(image)
    
    # Calculate processing time
    processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    # Save the annotated image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_filename = f"fall_detection_{timestamp}.jpg"
    result_path = os.path.join(results_dir, result_filename)
    cv2.imwrite(result_path, annotated_img)
    
    # Save to database
    _, result_image_data = cv2.imencode('.jpg', annotated_img)
    _, original_image_data = cv2.imencode('.jpg', image)
    
    db.save_detection(
        file_name=file.filename,
        file_type=file.content_type,
        fall_detected=fall_detected,
        confidence=float(fall_confidence),
        processing_time=processing_time,
        original_image_path="",  # We're storing the actual image in the database
        result_image_path=result_path,
        original_image_data=original_image_data.tobytes(),
        result_image_data=result_image_data.tobytes(),
        model_version=model  # Save the model version used
    )
    
    # Return results
    return {
        "fall_detected": fall_detected,
        "confidence": float(fall_confidence),
        "result_image": f"/results/{result_filename}",
        "processing_time": processing_time,
        "model_version": model
    }

@app.post("/process-video")
async def process_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model: str = Form("YOLOv8"),
):
    """
    Process a video file for fall detection
    
    Args:
        file: Video file to analyze
        model: YOLO model version to use (YOLOv8 or YOLOv8-P2)
    
    Returns:
        JSON response with job ID for tracking the processing
    """
    # Validate file
    if not file.content_type.startswith("video/"):
        return JSONResponse(
            status_code=400,
            content={"error": "Only video files are supported"}
        )
    
    # Save the uploaded file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = f"job_{timestamp}"
    video_path = os.path.join(uploads_dir, f"{job_id}_input.mp4")
    
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Create output path
    output_path = os.path.join(results_dir, f"{job_id}_output.mp4")
    results_path = os.path.join(results_dir, f"{job_id}_results.json")
    
    # Save job to database
    db.save_video_job(job_id, file.filename, model)
    
    # Process video in background
    background_tasks.add_task(
        process_video_task,
        video_path,
        output_path,
        results_path,
        job_id,
        model
    )
    
    return {
        "job_id": job_id,
        "status": "processing",
        "message": "Video processing started"
    }

@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """
    Get the status of a video processing job
    
    Args:
        job_id: ID of the job to check
    
    Returns:
        JSON response with job status and results if available
    """
    results_path = os.path.join(results_dir, f"{job_id}_results.json")
    output_path = os.path.join(results_dir, f"{job_id}_output.mp4")
    
    if os.path.exists(results_path):
        # Job completed
        with open(results_path, "r") as f:
            results = json.load(f)
        
        return {
            "job_id": job_id,
            "status": "completed",
            "results": results,
            "output_video": output_path if os.path.exists(output_path) else None
        }
    
    # Check if job is still processing
    input_path = os.path.join(uploads_dir, f"{job_id}_input.mp4")
    if os.path.exists(input_path):
        return {
            "job_id": job_id,
            "status": "processing",
            "message": "Video is still being processed"
        }
    
    # Job not found
    return JSONResponse(
        status_code=404,
        content={
            "job_id": job_id,
            "status": "not_found",
            "message": "Job not found"
        }
    )

def process_video_task(video_path: str, output_path: str, results_path: str, job_id: str, model_version: str):
    """
    Background task to process a video file
    
    Args:
        video_path: Path to the input video
        output_path: Path to save the output video
        results_path: Path to save the results JSON
        job_id: ID of the job
        model_version: YOLO model version to use
    """
    try:
        logger.info(f"Processing video: {job_id} with {model_version}")
        
        # Select the appropriate model based on the request
        if model_version == "YOLOv8":
            yolo_model = app.state.yolov8_model
        elif model_version == "YOLOv8-P2":
            yolo_model = app.state.yolov8p2_model
        else:
            raise Exception(f"Invalid model: {model_version}. Supported models: YOLOv8, YOLOv8-P2")
        
        # Process the video
        fall_events = yolo_model.process_video(video_path, output_path)
        
        # Update database with completed status and fall events
        db.update_video_job(job_id, "completed", output_path)
        db.save_fall_events(job_id, fall_events, model_version)
        
        # Save results to file system as well
        results = {
            "job_id": job_id,
            "fall_events": fall_events,
            "processed_at": datetime.now().isoformat(),
            "output_video": f"/results/{os.path.basename(output_path)}",
            "model_version": model_version
        }
        
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Video processing completed: {job_id}")
    except Exception as e:
        logger.error(f"Error processing video {job_id}: {str(e)}")
        
        # Update database with error status
        db.update_video_job(job_id, "error", error_message=str(e))
        
        # Save error results to file system
        error_results = {
            "job_id": job_id,
            "error": str(e),
            "processed_at": datetime.now().isoformat()
        }
        
        with open(results_path, "w") as f:
            json.dump(error_results, f, indent=2)
    finally:
        # Clean up input file
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except Exception as e:
            logger.error(f"Error cleaning up file {video_path}: {str(e)}")

# Add static file serving for results directory
app.mount("/results", StaticFiles(directory=results_dir), name="results")

# Add API endpoints for database access
@app.get("/history/detections")
async def get_detection_history(limit: int = Query(20, ge=1, le=100)):
    """
    Get history of image detections
    
    Args:
        limit: Maximum number of results to return
        
    Returns:
        List of detection results
    """
    detections = db.get_recent_detections(limit)
    
    # Convert paths to URLs
    for detection in detections:
        if detection["result_image_path"]:
            detection["result_image_url"] = f"/results/{os.path.basename(detection['result_image_path'])}"
    
    return {"detections": detections}

@app.get("/history/videos")
async def get_video_history(limit: int = Query(20, ge=1, le=100)):
    """
    Get history of video processing jobs
    
    Args:
        limit: Maximum number of results to return
        
    Returns:
        List of video processing jobs
    """
    jobs = db.get_recent_video_jobs(limit)
    
    # Convert paths to URLs
    for job in jobs:
        if job["output_video_path"]:
            job["output_video_url"] = f"/results/{os.path.basename(job['output_video_path'])}"
    
    return {"videos": jobs}

@app.get("/history/detection/{detection_id}")
async def get_detection_detail(detection_id: int = Path(...)):
    """
    Get details of a specific detection
    
    Args:
        detection_id: ID of the detection
        
    Returns:
        Detection details
    """
    detection = db.get_detection(detection_id)
    
    if not detection:
        return JSONResponse(
            status_code=404,
            content={"error": f"Detection {detection_id} not found"}
        )
    
    # Convert paths to URLs
    if detection["result_image_path"]:
        detection["result_image_url"] = f"/results/{os.path.basename(detection['result_image_path'])}"
    
    return detection

@app.get("/history/detection/{detection_id}/image/{image_type}")
async def get_detection_image(detection_id: int = Path(...), image_type: str = Path(...)):
    """
    Get the image for a detection
    
    Args:
        detection_id: ID of the detection
        image_type: Type of image (original or processed)
        
    Returns:
        Image file
    """
    if image_type not in ["original", "processed"]:
        return JSONResponse(
            status_code=400,
            content={"error": "Image type must be 'original' or 'processed'"}
        )
    
    detection = db.get_detection(detection_id)
    
    if not detection:
        return JSONResponse(
            status_code=404,
            content={"error": f"Detection {detection_id} not found"}
        )
    
    image_data_key = "original_image_data" if image_type == "original" else "result_image_data"
    
    if image_data_key not in detection or not detection[image_data_key]:
        return JSONResponse(
            status_code=404,
            content={"error": f"{image_type.capitalize()} image not found for detection {detection_id}"}
        )
    
    return Response(
        content=detection[image_data_key],
        media_type="image/jpeg"
    )

# Add static file serving for web interface
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

if __name__ == "__main__":
    # Direct execution of server.py
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
