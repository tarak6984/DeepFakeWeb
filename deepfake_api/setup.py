#!/usr/bin/env python3
"""
Setup script for Deepfake Detection API
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd):
    """Run a shell command"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    print(result.stdout)
    return True

def install_dependencies():
    """Install Python dependencies"""
    print("Installing Python dependencies...")
    
    # Update pip
    if not run_command(f"{sys.executable} -m pip install --upgrade pip"):
        return False
    
    # Install requirements
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt"):
        return False
    
    return True

def download_models():
    """Download and setup models"""
    print("Downloading models...")
    
    if not run_command(f"{sys.executable} download_models.py"):
        print("Warning: Model download failed. You can run this manually later.")
        return False
    
    return True

def create_directories():
    """Create necessary directories"""
    print("Creating directories...")
    
    directories = [
        "models/image",
        "models/audio", 
        "models/video",
        "models/multimodal",
        "uploads",
        "temp",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created: {directory}")
    
    return True

def setup_environment():
    """Setup environment variables"""
    print("Setting up environment...")
    
    env_content = """# Deepfake Detection API Configuration
DEEPFAKE_API_HOST=0.0.0.0
DEEPFAKE_API_PORT=8000
DEEPFAKE_API_DEBUG=false
DEEPFAKE_GPU_ENABLED=true
DEEPFAKE_DEVICE=auto
DEEPFAKE_MAX_FILE_SIZE=100
"""
    
    if not Path(".env").exists():
        with open(".env", "w") as f:
            f.write(env_content)
        print("Created .env file")
    
    return True

def main():
    """Main setup function"""
    print("Setting up Deepfake Detection API...")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        return False
    
    try:
        # Create directories
        if not create_directories():
            return False
        
        # Setup environment
        if not setup_environment():
            return False
        
        # Install dependencies
        if not install_dependencies():
            return False
        
        # Download models
        download_models()  # Don't fail if this doesn't work
        
        print("\n" + "=" * 50)
        print("Setup completed successfully!")
        print("\nTo start the API:")
        print("python main.py")
        print("\nOr use uvicorn:")
        print("uvicorn main:app --host 0.0.0.0 --port 8000")
        print("\nAPI will be available at: http://localhost:8000")
        print("Documentation at: http://localhost:8000/docs")
        
        return True
        
    except Exception as e:
        print(f"Setup failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)