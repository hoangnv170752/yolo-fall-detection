import sqlite3
import os
import base64
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple, Any, Union

# Configure logging
logger = logging.getLogger("fall-detection-api")

class FallDetectionDB:
    def __init__(self, db_path: str = None):
        """Initialize the database connection"""
        if db_path is None:
            # Default path is in the api directory
            db_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(db_dir, "fall_detection.db")
        
        self.db_path = db_path
        self._create_tables()
        logger.info(f"Database initialized at {db_path}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        return conn
    
    def _create_tables(self):
        """Create database tables if they don't exist"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Create detections table for storing detection results
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            file_name TEXT NOT NULL,
            file_type TEXT NOT NULL,
            fall_detected BOOLEAN NOT NULL,
            confidence REAL,
            processing_time REAL,
            original_image_path TEXT,
            result_image_path TEXT,
            model_version TEXT
        )
        ''')
        
        # Create images table for storing binary image data
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_id INTEGER NOT NULL,
            image_type TEXT NOT NULL,  -- 'original' or 'processed'
            image_data BLOB NOT NULL,
            FOREIGN KEY (detection_id) REFERENCES detections (id) ON DELETE CASCADE
        )
        ''')
        
        # Create video_jobs table for tracking video processing jobs
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS video_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT UNIQUE NOT NULL,
            timestamp TEXT NOT NULL,
            file_name TEXT NOT NULL,
            status TEXT NOT NULL,  -- 'processing', 'completed', 'error'
            output_video_path TEXT,
            error_message TEXT,
            model_version TEXT
        )
        ''')
        
        # Create fall_events table for storing fall events detected in videos
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS fall_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            frame_number INTEGER NOT NULL,
            confidence REAL NOT NULL,
            model_version TEXT,
            FOREIGN KEY (job_id) REFERENCES video_jobs (id) ON DELETE CASCADE
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_detection(self, 
                      file_name: str, 
                      file_type: str, 
                      fall_detected: bool, 
                      confidence: float,
                      processing_time: float,
                      original_image_path: str,
                      result_image_path: str,
                      original_image_data: bytes = None,
                      result_image_data: bytes = None,
                      model_version: str = None) -> int:
        """
        Save a detection result to the database
        
        Args:
            file_name: Name of the processed file
            file_type: Type of file (image/jpeg, etc.)
            fall_detected: Whether a fall was detected
            confidence: Confidence score of the detection
            processing_time: Time taken to process the image in ms
            original_image_path: Path to the original image
            result_image_path: Path to the processed image
            original_image_data: Binary data of the original image (optional)
            result_image_data: Binary data of the processed image (optional)
            
        Returns:
            The ID of the inserted detection
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        # Insert detection record
        cursor.execute('''
        INSERT INTO detections (
            timestamp, file_name, file_type, fall_detected, confidence, 
            processing_time, original_image_path, result_image_path, model_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            timestamp, file_name, file_type, fall_detected, confidence,
            processing_time, original_image_path, result_image_path, model_version
        ))
        
        detection_id = cursor.lastrowid
        
        # Save original image data if provided
        if original_image_data:
            cursor.execute('''
            INSERT INTO images (detection_id, image_type, image_data)
            VALUES (?, ?, ?)
            ''', (detection_id, 'original', original_image_data))
        
        # Save result image data if provided
        if result_image_data:
            cursor.execute('''
            INSERT INTO images (detection_id, image_type, image_data)
            VALUES (?, ?, ?)
            ''', (detection_id, 'processed', result_image_data))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Saved detection ID {detection_id} for file {file_name}")
        return detection_id
    
    def save_video_job(self, job_id: str, file_name: str, model_version: str = None) -> int:
        """
        Save a new video processing job
        
        Args:
            job_id: Unique ID for the job
            file_name: Name of the video file
            
        Returns:
            The ID of the inserted job
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        cursor.execute('''
        INSERT INTO video_jobs (job_id, timestamp, file_name, status, model_version)
        VALUES (?, ?, ?, ?, ?)
        ''', (job_id, timestamp, file_name, 'processing', model_version))
        
        job_db_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        logger.info(f"Saved video job {job_id} for file {file_name}")
        return job_db_id
    
    def update_video_job(self, job_id: str, status: str, output_video_path: str = None, error_message: str = None) -> bool:
        """
        Update the status of a video processing job
        
        Args:
            job_id: Unique ID for the job
            status: New status ('completed' or 'error')
            output_video_path: Path to the output video (if completed)
            error_message: Error message (if status is 'error')
            
        Returns:
            True if the job was updated, False otherwise
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if status == 'completed':
            cursor.execute('''
            UPDATE video_jobs
            SET status = ?, output_video_path = ?
            WHERE job_id = ?
            ''', (status, output_video_path, job_id))
        elif status == 'error':
            cursor.execute('''
            UPDATE video_jobs
            SET status = ?, error_message = ?
            WHERE job_id = ?
            ''', (status, error_message, job_id))
        else:
            cursor.execute('''
            UPDATE video_jobs
            SET status = ?
            WHERE job_id = ?
            ''', (status, job_id))
        
        updated = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        logger.info(f"Updated video job {job_id} status to {status}")
        return updated
    
    def save_fall_events(self, job_id: str, fall_events: List[Dict[str, Any]], model_version: str = None) -> bool:
        """
        Save fall events detected in a video
        
        Args:
            job_id: Unique ID for the job
            fall_events: List of fall events with timestamp, frame_number, and confidence
            
        Returns:
            True if events were saved, False otherwise
        """
        if not fall_events:
            return True
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get the database ID for the job
        cursor.execute('SELECT id FROM video_jobs WHERE job_id = ?', (job_id,))
        result = cursor.fetchone()
        
        if not result:
            logger.error(f"Job {job_id} not found in database")
            conn.close()
            return False
        
        job_db_id = result['id']
        
        # Insert fall events
        for event in fall_events:
            cursor.execute('''
            INSERT INTO fall_events (job_id, timestamp, frame_number, confidence, model_version)
            VALUES (?, ?, ?, ?, ?)
            ''', (
                job_db_id,
                event.get('timestamp', datetime.now().isoformat()),
                event.get('frame', 0),
                event.get('confidence', 0.0),
                model_version
            ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Saved {len(fall_events)} fall events for job {job_id}")
        return True
    
    def get_recent_detections(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get recent detection results
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List of detection results
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT * FROM detections
        ORDER BY timestamp DESC
        LIMIT ?
        ''', (limit,))
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return results
    
    def get_detection(self, detection_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a specific detection by ID
        
        Args:
            detection_id: ID of the detection
            
        Returns:
            Detection data or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM detections WHERE id = ?', (detection_id,))
        result = cursor.fetchone()
        
        if result:
            detection = dict(result)
            
            # Get images if available
            cursor.execute('''
            SELECT image_type, image_data FROM images
            WHERE detection_id = ?
            ''', (detection_id,))
            
            images = cursor.fetchall()
            for img in images:
                if img['image_type'] == 'original':
                    detection['original_image_data'] = img['image_data']
                elif img['image_type'] == 'processed':
                    detection['result_image_data'] = img['image_data']
            
            conn.close()
            return detection
        
        conn.close()
        return None
    
    def get_video_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a video job by ID
        
        Args:
            job_id: Unique ID for the job
            
        Returns:
            Job data or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM video_jobs WHERE job_id = ?', (job_id,))
        job = cursor.fetchone()
        
        if not job:
            conn.close()
            return None
        
        job_dict = dict(job)
        job_db_id = job_dict['id']
        
        # Get fall events for this job
        cursor.execute('''
        SELECT * FROM fall_events
        WHERE job_id = ?
        ORDER BY frame_number
        ''', (job_db_id,))
        
        fall_events = [dict(row) for row in cursor.fetchall()]
        job_dict['fall_events'] = fall_events
        
        conn.close()
        return job_dict
    
    def get_recent_video_jobs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get recent video jobs
        
        Args:
            limit: Maximum number of jobs to return
            
        Returns:
            List of video jobs
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT * FROM video_jobs
        ORDER BY timestamp DESC
        LIMIT ?
        ''', (limit,))
        
        jobs = [dict(row) for row in cursor.fetchall()]
        
        # Get fall events for each job
        for job in jobs:
            cursor.execute('''
            SELECT * FROM fall_events
            WHERE job_id = ?
            ORDER BY frame_number
            ''', (job['id'],))
            
            job['fall_events'] = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return jobs
