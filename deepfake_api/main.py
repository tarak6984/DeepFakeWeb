#!/usr/bin/env python3
"""
Unified Deepfake Detection API
A powerful local alternative to Reality Defender with support for images, audio, and video
"""

# Production-ready logging configuration - suppress harmless warnings
import os
import warnings

# Suppress TensorFlow INFO messages and oneDNN warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Only show warnings and errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Deterministic results

# Suppress Pydantic and other harmless warnings
warnings.filterwarnings("ignore", message=".*repr.*attribute.*Field.*")
warnings.filterwarnings("ignore", message=".*frozen.*attribute.*Field.*")
warnings.filterwarnings("ignore", message=".*Mapping deprecated model name.*")

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import logging
import yaml
import os
import tempfile
import shutil
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Union
from pathlib import Path
import asyncio
import aiofiles
import time
from functools import lru_cache
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
import threading

# Import detectors
from detectors.image_detector import ImageDeepfakeDetector
from detectors.audio_detector import AudioDeepfakeDetector
from detectors.video_detector import VideoDeepfakeDetector

# Setup logging with clean production output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress noisy library warnings
logging.getLogger("timm").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Load configuration
def load_config():
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    else:
        # Default configuration
        return {
            'api': {
                'host': '0.0.0.0',
                'port': 8000,
                'debug': False,
                'cors_origins': ["http://localhost:3000", "http://127.0.0.1:3000"],
                'max_file_size': 100,  # MB
                'upload_dir': 'uploads',
                'temp_dir': 'temp'
            },
            'processing': {
                'gpu_enabled': True,
                'device': 'auto'
            }
        }

config = load_config()

# Initialize detectors globally (for efficiency)
image_detector = None
audio_detector = None
video_detector = None

# Storage for analysis results (in production, use a database)
analysis_results = {}

# High-performance caching system
result_cache = {}
cache_dir = Path("cache")
cache_dir.mkdir(exist_ok=True)

def get_file_hash(file_content: bytes) -> str:
    """Generate MD5 hash of file content for caching"""
    return hashlib.md5(file_content).hexdigest()

def get_cached_result(file_hash: str, file_type: str) -> Optional[Dict]:
    """Get cached result if available - DISABLED FOR FRESH ANALYSIS"""
    # Caching disabled - always return None for fresh analysis
    return None
        
    cache_file = cache_dir / f"{file_hash}_{file_type}.json"
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cached = json.load(f)
                # Check if cache is still valid (24 hours)
                if time.time() - cached['cached_at'] < 86400:
                    logger.info(f"🚀 Cache HIT for {file_hash[:8]}... - INSTANT RESULT!")
                    return cached['result']
        except:
            pass
    return None

def save_to_cache(file_hash: str, file_type: str, result: Dict):
    """Save result to cache - DISABLED FOR FRESH ANALYSIS"""
    # Caching disabled - skip saving to cache
    return
        
    try:
        cache_file = cache_dir / f"{file_hash}_{file_type}.json"
        cache_data = {
            'result': result,
            'cached_at': time.time(),
            'file_hash': file_hash
        }
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        logger.info(f"💾 Saved to cache: {file_hash[:8]}...")
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize detectors on startup and cleanup on shutdown"""
    global image_detector, audio_detector, video_detector
    
    # Startup
    try:
        logger.info("Initializing deepfake detectors...")
        
        device = config['processing']['device']
        
        # Initialize detectors
        image_detector = ImageDeepfakeDetector(device=device)
        logger.info("Image detector initialized")
        
        audio_detector = AudioDeepfakeDetector(device=device)
        logger.info("Audio detector initialized")
        
        video_detector = VideoDeepfakeDetector(device=device)
        logger.info("Video detector initialized")
        
        logger.info("All detectors initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize detectors: {e}")
    
    yield
    
    # Shutdown (cleanup if needed)
    logger.info("Shutting down deepfake detection API")

# Create FastAPI app
app = FastAPI(
    title="Deepfake Detection API",
    description="Advanced AI-powered deepfake detection for images, audio, and video files",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config['api']['cors_origins'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
upload_dir = Path(config['api']['upload_dir'])
temp_dir = Path(config['api']['temp_dir'])
upload_dir.mkdir(exist_ok=True)
temp_dir.mkdir(exist_ok=True)

def get_file_type(filename: str) -> str:
    """Determine file type based on extension"""
    ext = Path(filename).suffix.lower()
    
    # Image extensions
    if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff', '.tif']:
        return 'image'
    
    # Audio extensions
    elif ext in ['.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.wma']:
        return 'audio'
    
    # Video extensions
    elif ext in ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv', '.m4v']:
        return 'video'
    
    else:
        return 'unknown'

def validate_file_size(file: UploadFile) -> bool:
    """Check if file size is within limits"""
    max_size = config['api']['max_file_size'] * 1024 * 1024  # Convert MB to bytes
    
    # Get file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    
    return file_size <= max_size

async def save_upload_file(file: UploadFile) -> tuple[str, str, bytes]:
    """Save uploaded file and return path, hash, and content"""
    # Read file content once
    content = await file.read()
    file_hash = get_file_hash(content)
    
    # Create unique filename
    file_id = str(uuid.uuid4())
    file_ext = Path(file.filename).suffix
    file_path = upload_dir / f"{file_id}{file_ext}"
    
    # Save file
    async with aiofiles.open(file_path, 'wb') as f:
        await f.write(content)
    
    return str(file_path), file_hash, content

async def analyze_file(file_path: str, file_type: str, analysis_id: str, file_hash: str = None):
    """⚡ SUPER FAST analyze file with caching and optimization"""
    start_time = time.time()
    
    try:
        # 🚀 STEP 1: Check cache first for instant results
        if file_hash:
            cached_result = get_cached_result(file_hash, file_type)
            if cached_result:
                cached_result.update({
                    'analysis_id': analysis_id,
                    'timestamp': datetime.utcnow().isoformat(),
                    'status': 'completed',
                    'cache_hit': True,
                    'processing_time': time.time() - start_time
                })
                analysis_results[analysis_id] = cached_result
                logger.info(f"⚡ INSTANT CACHE HIT {analysis_id}: {cached_result.get('prediction', 'unknown')} ({cached_result.get('confidence', 0.0):.3f}) - {(time.time() - start_time)*1000:.1f}ms")
                return
        
        logger.info(f"🔍 Starting fresh analysis {analysis_id} for {file_type}: {file_path}")
        
        # 🚀 STEP 2: Run detection with optimized settings
        if file_type == 'image':
            result = image_detector.detect(file_path)
        elif file_type == 'audio':
            result = audio_detector.detect(file_path)
        elif file_type == 'video':
            result = video_detector.detect(file_path)
        else:
            result = {
                'error': f'Unsupported file type: {file_type}',
                'prediction': 'unknown',
                'confidence': 0.0
            }
        
        processing_time = time.time() - start_time
        
        # Add metadata
        result.update({
            'analysis_id': analysis_id,
            'file_type': file_type,
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'completed',
            'cache_hit': False,
            'processing_time': processing_time
        })
        
        # Store result
        analysis_results[analysis_id] = result
        
        # 🚀 STEP 3: Cache for future speed
        if file_hash and 'error' not in result:
            save_to_cache(file_hash, file_type, result)
        
        logger.info(f"✅ Analysis {analysis_id} completed: {result.get('prediction', 'unknown')} ({result.get('confidence', 0.0):.3f}) - {processing_time*1000:.1f}ms")
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"❌ Analysis {analysis_id} failed: {e} - {processing_time*1000:.1f}ms")
        analysis_results[analysis_id] = {
            'analysis_id': analysis_id,
            'error': str(e),
            'prediction': 'unknown',
            'confidence': 0.0,
            'status': 'failed',
            'timestamp': datetime.utcnow().isoformat(),
            'processing_time': processing_time
        }
    
    finally:
        # Clean up file
        try:
            os.unlink(file_path)
        except:
            pass

# API Routes

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Deepfake Detection API",
        "version": "1.0.0",
        "description": "Advanced AI-powered deepfake detection for images, audio, and video",
        "endpoints": {
            "upload": "/api/upload",
            "result": "/api/result/{analysis_id}",
            "batch": "/api/batch",
            "health": "/api/health"
        },
        "supported_formats": {
            "images": ["jpg", "jpeg", "png", "bmp", "gif", "webp", "tiff"],
            "audio": ["mp3", "wav", "flac", "m4a", "aac", "ogg", "wma"],
            "video": ["mp4", "avi", "mov", "wmv", "flv", "webm", "mkv", "m4v"]
        }
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "detectors": {
            "image": image_detector is not None,
            "audio": audio_detector is not None,
            "video": video_detector is not None
        }
    }

@app.post("/api/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload and analyze a single file"""
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file size
    if not validate_file_size(file):
        raise HTTPException(
            status_code=413, 
            detail=f"File too large. Maximum size: {config['api']['max_file_size']}MB"
        )
    
    # Determine file type
    file_type = get_file_type(file.filename)
    if file_type == 'unknown':
        raise HTTPException(status_code=400, detail="Unsupported file format")
    
    try:
        start_time = time.time()
        
        # Save file and get hash
        file_path, file_hash, file_content = await save_upload_file(file)
        
        # Generate analysis ID
        analysis_id = str(uuid.uuid4())
        
        # 🚀 SUPER SPEED: Check cache first for instant results!
        cached_result = get_cached_result(file_hash, file_type)
        if cached_result:
            # INSTANT RESULT from cache!
            cached_result.update({
                'analysis_id': analysis_id,
                'filename': file.filename,
                'file_type': file_type,
                'timestamp': datetime.utcnow().isoformat(),
                'status': 'completed',
                'cache_hit': True,
                'processing_time': time.time() - start_time
            })
            analysis_results[analysis_id] = cached_result
            
            # Clean up uploaded file since we have cached result
            try:
                os.unlink(file_path)
            except:
                pass
                
            logger.info(f"⚡ INSTANT CACHE RESULT for {file.filename}: {cached_result.get('prediction')} ({cached_result.get('confidence', 0.0):.3f})")
            
            return {
                "analysis_id": analysis_id,
                "filename": file.filename,
                "file_type": file_type,
                "status": "completed",
                "cache_hit": True,
                "message": "⚡ Instant result from cache!"
            }
        
        # Store initial result for processing
        analysis_results[analysis_id] = {
            'analysis_id': analysis_id,
            'filename': file.filename,
            'file_type': file_type,
            'status': 'processing',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Start background analysis with file hash for caching
        background_tasks.add_task(analyze_file, file_path, file_type, analysis_id, file_hash)
        
        return {
            "analysis_id": analysis_id,
            "filename": file.filename,
            "file_type": file_type,
            "status": "processing",
            "message": "File uploaded successfully. Analysis in progress."
        }
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/api/result/{analysis_id}")
async def get_result(analysis_id: str):
    """Get analysis result by ID"""
    
    if analysis_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    result = analysis_results[analysis_id]
    
    # Clean up old results if completed
    if result.get('status') in ['completed', 'failed']:
        # Keep result for some time, then clean up in production
        pass
    
    return result

@app.post("/api/batch")
async def batch_upload(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """Upload and analyze multiple files"""
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch")
    
    batch_id = str(uuid.uuid4())
    analysis_ids = []
    
    for file in files:
        # Validate each file
        if not file.filename:
            continue
            
        if not validate_file_size(file):
            continue
            
        file_type = get_file_type(file.filename)
        if file_type == 'unknown':
            continue
        
        try:
            # Save file
            file_path = await save_upload_file(file)
            
            # Generate analysis ID
            analysis_id = str(uuid.uuid4())
            analysis_ids.append(analysis_id)
            
            # Store initial result
            analysis_results[analysis_id] = {
                'analysis_id': analysis_id,
                'batch_id': batch_id,
                'filename': file.filename,
                'file_type': file_type,
                'status': 'processing',
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Start background analysis
            background_tasks.add_task(analyze_file, file_path, file_type, analysis_id)
            
        except Exception as e:
            logger.error(f"Failed to process file {file.filename}: {e}")
            continue
    
    return {
        "batch_id": batch_id,
        "analysis_ids": analysis_ids,
        "status": "processing",
        "message": f"Batch upload successful. {len(analysis_ids)} files being analyzed."
    }

@app.get("/api/batch/{batch_id}")
async def get_batch_results(batch_id: str):
    """Get results for all analyses in a batch"""
    
    batch_results = []
    for analysis_id, result in analysis_results.items():
        if result.get('batch_id') == batch_id:
            batch_results.append(result)
    
    if not batch_results:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    # Calculate batch status
    statuses = [r.get('status', 'unknown') for r in batch_results]
    if all(s in ['completed', 'failed'] for s in statuses):
        batch_status = 'completed'
    elif any(s == 'processing' for s in statuses):
        batch_status = 'processing'
    else:
        batch_status = 'unknown'
    
    return {
        "batch_id": batch_id,
        "status": batch_status,
        "total_files": len(batch_results),
        "completed": len([r for r in batch_results if r.get('status') == 'completed']),
        "failed": len([r for r in batch_results if r.get('status') == 'failed']),
        "results": batch_results
    }

# Reality Defender compatible endpoints (for easy replacement)
@app.post("/api/files/aws-presigned")
async def get_signed_url(request: dict):
    """Reality Defender compatible endpoint - not needed for local API"""
    return {
        "message": "Local API does not require signed URLs. Use /api/upload instead.",
        "local_endpoint": "/api/upload"
    }

@app.get("/api/media/users/{analysis_id}")
async def get_media_result(analysis_id: str):
    """Reality Defender compatible endpoint"""
    return await get_result(analysis_id)

# Error handlers
@app.exception_handler(413)
async def payload_too_large_handler(request, exc):
    return JSONResponse(
        status_code=413,
        content={
            "error": "File too large",
            "max_size_mb": config['api']['max_file_size']
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "Please try again later"
        }
    )

if __name__ == "__main__":
    # Create initial directories
    os.makedirs("models/image", exist_ok=True)
    os.makedirs("models/audio", exist_ok=True)
    os.makedirs("models/video", exist_ok=True)
    
    # Run server
    uvicorn.run(
        "main:app",
        host=config['api']['host'],
        port=config['api']['port'],
        reload=config['api']['debug'],
        log_level="info"
    )