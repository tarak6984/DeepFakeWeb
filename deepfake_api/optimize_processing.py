#!/usr/bin/env python3
"""
Dynamic Processing Optimization Script
Automatically adjusts processing settings based on file size and type
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ProcessingOptimizer:
    """Optimizes processing settings based on file characteristics"""
    
    def __init__(self):
        # File size thresholds in MB
        self.SMALL_FILE = 5    # < 5MB
        self.MEDIUM_FILE = 25  # 5-25MB  
        self.LARGE_FILE = 100  # 25-100MB
        self.HUGE_FILE = 500   # > 100MB
    
    def get_file_size_mb(self, file_path: str) -> float:
        """Get file size in MB"""
        try:
            size_bytes = os.path.getsize(file_path)
            return size_bytes / (1024 * 1024)
        except OSError:
            return 0
    
    def get_optimal_settings(self, file_path: str, file_type: str) -> dict:
        """Get optimal processing settings based on file size and type"""
        file_size_mb = self.get_file_size_mb(file_path)
        
        settings = {
            'ultra_fast_mode': False,
            'fast_mode': True,
            'batch_size': 8,
            'max_segments': 10,
            'frames_per_analysis': 30,
            'skip_vad': False,
            'max_duration': None,
            'resize_resolution': None
        }
        
        # Optimize based on file size
        if file_size_mb < self.SMALL_FILE:
            # Small files - standard processing
            settings.update({
                'ultra_fast_mode': False,
                'fast_mode': False,
                'batch_size': 16,
                'max_segments': 15,
                'frames_per_analysis': 50,
                'skip_vad': False
            })
            
        elif file_size_mb < self.MEDIUM_FILE:
            # Medium files - fast mode
            settings.update({
                'ultra_fast_mode': False,
                'fast_mode': True,
                'batch_size': 12,
                'max_segments': 10,
                'frames_per_analysis': 30,
                'skip_vad': True
            })
            
        elif file_size_mb < self.LARGE_FILE:
            # Large files - ultra fast mode
            settings.update({
                'ultra_fast_mode': True,
                'fast_mode': True,
                'batch_size': 8,
                'max_segments': 5,
                'frames_per_analysis': 15,
                'skip_vad': True,
                'max_duration': 60,  # Limit to 60 seconds
                'resize_resolution': 1280
            })
            
        else:
            # Huge files - maximum optimization
            settings.update({
                'ultra_fast_mode': True,
                'fast_mode': True,
                'batch_size': 4,
                'max_segments': 2,
                'frames_per_analysis': 10,
                'skip_vad': True,
                'max_duration': 30,  # Limit to 30 seconds
                'resize_resolution': 720
            })
        
        # File type specific optimizations
        if file_type.startswith('video/'):
            settings['skip_frames'] = min(5, max(1, int(file_size_mb / 10)))
            if file_size_mb > self.LARGE_FILE:
                settings['frame_sampling'] = 'smart'
                
        elif file_type.startswith('audio/'):
            if file_size_mb > self.MEDIUM_FILE:
                settings['duration'] = min(3, max(1, 10 / file_size_mb))
                
        elif file_type.startswith('image/'):
            if file_size_mb > 10:  # Large images
                settings['fast_resize'] = True
                settings['resize_resolution'] = min(1920, max(720, 2000 - file_size_mb * 20))
        
        logger.info(f"Optimized settings for {file_size_mb:.1f}MB {file_type}: {settings}")
        return settings
    
    def estimate_processing_time(self, file_path: str, file_type: str, settings: dict) -> float:
        """Estimate processing time in seconds"""
        file_size_mb = self.get_file_size_mb(file_path)
        
        # Base time estimates (seconds per MB)
        base_times = {
            'image/': 0.1,
            'audio/': 0.5,
            'video/': 2.0
        }
        
        media_type = next((k for k in base_times.keys() if file_type.startswith(k)), 'image/')
        base_time = base_times[media_type]
        
        # Apply speed multipliers
        speed_multiplier = 1.0
        if settings.get('ultra_fast_mode'):
            speed_multiplier *= 0.3  # 70% faster
        if settings.get('fast_mode'):
            speed_multiplier *= 0.6  # 40% faster
        if settings.get('skip_vad'):
            speed_multiplier *= 0.8  # 20% faster
        
        estimated_time = file_size_mb * base_time * speed_multiplier
        return max(1, estimated_time)  # Minimum 1 second

# Global optimizer instance
optimizer = ProcessingOptimizer()

def optimize_for_file(file_path: str, file_type: str) -> dict:
    """Convenience function to get optimal settings for a file"""
    return optimizer.get_optimal_settings(file_path, file_type)

def estimate_time(file_path: str, file_type: str, settings: dict = None) -> float:
    """Convenience function to estimate processing time"""
    if settings is None:
        settings = optimize_for_file(file_path, file_type)
    return optimizer.estimate_processing_time(file_path, file_type, settings)