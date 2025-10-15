import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import logging
import yaml
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
import moviepy.editor as mp
import tempfile
import os
import time
from .image_detector import FaceExtractor, EfficientNetDeepfakeDetector

logger = logging.getLogger(__name__)

class Video3DCNN(nn.Module):
    """3D CNN for temporal deepfake detection"""
    
    def __init__(self, num_classes: int = 2, input_channels: int = 3, temporal_length: int = 16, fc_features: int = 512):
        super().__init__()
        self.temporal_length = temporal_length
        self.fc_features = fc_features
        
        # 3D Convolutional layers
        self.conv3d1 = nn.Conv3d(input_channels, 64, kernel_size=(3, 7, 7), 
                                stride=(1, 2, 2), padding=(1, 3, 3))
        self.bn1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        self.conv3d2 = nn.Conv3d(64, 128, kernel_size=(3, 5, 5), 
                                stride=(1, 1, 1), padding=(1, 2, 2))
        self.bn2 = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 2, 2), padding=(0, 1, 1))
        
        self.conv3d3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), 
                                stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(256)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 2, 2), padding=(0, 1, 1))
        
        # Adaptive feature layer - can be 256 or 512
        if fc_features == 256:
            # Simpler architecture for 256 features
            self.feature_layer = nn.Sequential(
                nn.AdaptiveAvgPool3d(1)
            )
            self.feature_dim = 256
        else:
            # Full architecture for 512 features
            self.conv3d4 = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), 
                                    stride=(1, 1, 1), padding=(1, 1, 1))
            self.bn4 = nn.BatchNorm3d(512)
            self.pool4 = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 2, 2), padding=(0, 1, 1))
            self.feature_layer = nn.Sequential(
                nn.AdaptiveAvgPool3d(1)
            )
            self.feature_dim = 512
        
        # Classifier
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(self.feature_dim, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, channels, temporal_length, height, width)
        x = F.relu(self.bn1(self.conv3d1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv3d2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3d3(x)))
        x = self.pool3(x)
        
        # Adaptive processing based on feature dimension
        if self.feature_dim == 512:
            x = F.relu(self.bn4(self.conv3d4(x)))
            x = self.pool4(x)
        
        # Feature extraction and pooling
        x = self.feature_layer(x)
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

class TemporalConsistencyDetector(nn.Module):
    """Detect temporal inconsistencies in video frames"""
    
    def __init__(self, feature_dim: int = 512, num_classes: int = 2):
        super().__init__()
        
        # Feature extractor (using pre-trained CNN)
        import timm
        self.feature_extractor = timm.create_model('efficientnet_b4', pretrained=True, num_classes=0)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=256, 
                           num_layers=2, batch_first=True, dropout=0.3)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, dropout=0.1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, channels, height, width)
        batch_size, seq_len, c, h, w = x.shape
        
        # Extract features for each frame
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.feature_extractor(x)  # (batch_size * seq_len, feature_dim)
        features = features.view(batch_size, seq_len, -1)  # (batch_size, seq_len, feature_dim)
        
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(features)  # (batch_size, seq_len, hidden_size)
        
        # Attention
        lstm_out = lstm_out.transpose(0, 1)  # (seq_len, batch_size, hidden_size)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = attn_out.transpose(0, 1)  # (batch_size, seq_len, hidden_size)
        
        # Global temporal pooling
        temporal_features = torch.mean(attn_out, dim=1)  # (batch_size, hidden_size)
        
        # Classification
        output = self.classifier(temporal_features)
        
        return output

class VideoProcessor:
    """Video processing utilities"""
    
    def __init__(self, target_fps: int = 25, max_frames: int = 300):
        self.target_fps = target_fps
        self.max_frames = max_frames
        self.face_extractor = FaceExtractor()
        
        # Frame preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def extract_frames(self, video_path: str, sampling_strategy: str = "uniform") -> Optional[np.ndarray]:
        """Extract frames from video"""
        try:
            # Load video
            video = mp.VideoFileClip(video_path)
            
            # Resample to target FPS if needed
            if video.fps != self.target_fps:
                video = video.set_fps(self.target_fps)
            
            # Calculate frame indices to extract
            total_frames = int(video.duration * self.target_fps)
            
            if total_frames <= self.max_frames:
                frame_indices = list(range(0, total_frames))
            else:
                if sampling_strategy == "uniform":
                    frame_indices = np.linspace(0, total_frames-1, self.max_frames, dtype=int)
                elif sampling_strategy == "random":
                    frame_indices = sorted(np.random.choice(total_frames, self.max_frames, replace=False))
                else:
                    # Take first max_frames frames
                    frame_indices = list(range(self.max_frames))
            
            # Extract frames with proper cleanup
            frames = []
            try:
                for frame_idx in frame_indices:
                    time_point = frame_idx / self.target_fps
                    if time_point <= video.duration:
                        frame = video.get_frame(time_point)
                        frames.append(frame)
            finally:
                # Ensure video is always closed
                try:
                    video.close()
                except:
                    pass
            
            if frames:
                return np.array(frames)
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting frames from {video_path}: {e}")
            return None
    
    def extract_face_sequences(self, frames: np.ndarray) -> List[np.ndarray]:
        """Extract face sequences from video frames"""
        face_sequences = []
        
        try:
            all_face_crops = []
            
            for frame in frames:
                faces = self.face_extractor.extract_faces(frame)
                
                if faces:
                    # Use the largest face if multiple detected
                    largest_face = max(faces, key=lambda x: x.shape[0] * x.shape[1])
                    all_face_crops.append(largest_face)
                else:
                    # Use whole frame if no face detected
                    all_face_crops.append(frame)
            
            if all_face_crops:
                face_sequences.append(np.array(all_face_crops))
            
            return face_sequences
            
        except Exception as e:
            logger.error(f"Error extracting face sequences: {e}")
            return []
    
    def preprocess_for_3d_cnn(self, face_sequence: np.ndarray, temporal_length: int = 16) -> torch.Tensor:
        """Preprocess face sequence for 3D CNN"""
        try:
            # Ensure we have enough frames
            if len(face_sequence) < temporal_length:
                # Repeat frames to reach temporal_length
                repeat_factor = temporal_length // len(face_sequence) + 1
                face_sequence = np.tile(face_sequence, (repeat_factor, 1, 1, 1))
            
            # Take exactly temporal_length frames
            face_sequence = face_sequence[:temporal_length]
            
            # Apply transforms to each frame
            processed_frames = []
            for frame in face_sequence:
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                processed_frame = self.transform(frame)
                processed_frames.append(processed_frame)
            
            # Stack into tensor: (temporal_length, channels, height, width)
            video_tensor = torch.stack(processed_frames)
            
            # Rearrange to: (channels, temporal_length, height, width)
            video_tensor = video_tensor.permute(1, 0, 2, 3)
            
            return video_tensor.unsqueeze(0)  # Add batch dimension
            
        except Exception as e:
            logger.error(f"Error preprocessing for 3D CNN: {e}")
            return None
    
    def preprocess_for_temporal_consistency(self, face_sequence: np.ndarray, sequence_length: int = 30) -> torch.Tensor:
        """Preprocess face sequence for temporal consistency analysis"""
        try:
            # Sample frames uniformly
            if len(face_sequence) > sequence_length:
                indices = np.linspace(0, len(face_sequence)-1, sequence_length, dtype=int)
                face_sequence = face_sequence[indices]
            elif len(face_sequence) < sequence_length:
                # Pad sequence by repeating last frame
                padding_needed = sequence_length - len(face_sequence)
                last_frame = face_sequence[-1:] if len(face_sequence) > 0 else np.zeros((1, 224, 224, 3))
                padding = np.repeat(last_frame, padding_needed, axis=0)
                face_sequence = np.concatenate([face_sequence, padding], axis=0)
            
            # Apply transforms to each frame
            processed_frames = []
            for frame in face_sequence:
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                processed_frame = self.transform(frame)
                processed_frames.append(processed_frame)
            
            # Stack into tensor: (sequence_length, channels, height, width)
            sequence_tensor = torch.stack(processed_frames)
            
            return sequence_tensor.unsqueeze(0)  # Add batch dimension
            
        except Exception as e:
            logger.error(f"Error preprocessing for temporal consistency: {e}")
            return None

class VideoDeepfakeDetector:
    """Advanced video deepfake detection system"""
    
    def __init__(self, models_dir: str = "models/video", device: str = "auto"):
        self.models_dir = Path(models_dir)
        self.device = self._setup_device(device)
        self.video_processor = VideoProcessor()
        
        # Load config for thresholds
        self.config = self._load_config()
        self.confidence_threshold = self.config.get('models', {}).get('video', {}).get('confidence_threshold', 0.5)
        
        self.models = {}
        self.load_models()
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup compute device"""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)
    
    def _load_config(self):
        """Load configuration"""
        config_path = Path("config.yaml")
        if config_path.exists():
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        return {}
    
    def load_models(self):
        """Load pre-trained video deepfake detection models"""
        try:
            # Load 3D CNN model with adaptive architecture
            if (self.models_dir / "video_3d_cnn.pth").exists():
                try:
                    # First, check what fc layer size is expected by loading the state dict
                    state_dict = torch.load(
                        self.models_dir / "video_3d_cnn.pth", 
                        map_location=self.device,
                        weights_only=True
                    )
                    
                    # Check fc layer dimensions to determine architecture
                    if 'fc.weight' in state_dict:
                        fc_input_size = state_dict['fc.weight'].shape[1]
                        logger.info(f"🔧 Detected FC input size: {fc_input_size}")
                        
                        # Create model with correct architecture
                        self.models['3d_cnn'] = Video3DCNN(fc_features=fc_input_size)
                        
                        # Load with proper architecture match
                        self.models['3d_cnn'].load_state_dict(state_dict, strict=False)
                        self.models['3d_cnn'].to(self.device)
                        self.models['3d_cnn'].eval()
                        logger.info(f"✅ Loaded 3D CNN model with {fc_input_size}-feature architecture!")
                    else:
                        # Fallback to default architecture
                        self.models['3d_cnn'] = Video3DCNN()
                        self.models['3d_cnn'].load_state_dict(state_dict, strict=False)
                        self.models['3d_cnn'].to(self.device)
                        self.models['3d_cnn'].eval()
                        logger.info("✅ Loaded 3D CNN model with default architecture")
                        
                except Exception as e:
                    logger.warning(f"Failed to load 3D CNN checkpoint: {e}")
                    logger.info("🔄 Trying 256-feature architecture...")
                    try:
                        # Try 256-feature architecture
                        self.models['3d_cnn'] = Video3DCNN(fc_features=256)
                        state_dict = torch.load(
                            self.models_dir / "video_3d_cnn.pth", 
                            map_location=self.device,
                            weights_only=True
                        )
                        self.models['3d_cnn'].load_state_dict(state_dict, strict=False)
                        self.models['3d_cnn'].to(self.device)
                        self.models['3d_cnn'].eval()
                        logger.info("✅ Loaded 3D CNN model with 256-feature architecture!")
                    except:
                        logger.info("⚠️  Initializing 3D CNN with random weights...")
                        self.models['3d_cnn'] = Video3DCNN()
                        self.models['3d_cnn'].to(self.device)
                        self.models['3d_cnn'].eval()
            elif (self.models_dir / "faceforensics_config.json").exists():
                with open(self.models_dir / "faceforensics_config.json", "r") as f:
                    config = json.load(f)
                
                self.models['3d_cnn'] = Video3DCNN(
                    num_classes=config.get('num_classes', 2),
                    temporal_length=config.get('temporal_length', 16)
                )
                self.models['3d_cnn'].to(self.device)
                self.models['3d_cnn'].eval()
                logger.info("Loaded 3D CNN model (config only)")
            
            # Load Temporal Consistency model
            if (self.models_dir / "temporal_consistency.pth").exists():
                try:
                    from .video_models import TemporalConsistencyModel
                    self.models['temporal_consistency'] = TemporalConsistencyModel()
                    state_dict = torch.load(
                        self.models_dir / "temporal_consistency.pth", 
                        map_location=self.device,
                        weights_only=True
                    )
                    self.models['temporal_consistency'].load_state_dict(state_dict, strict=False)
                    self.models['temporal_consistency'].to(self.device)
                    self.models['temporal_consistency'].eval()
                    logger.info("Loaded Temporal Consistency model with trained weights")
                except Exception as e:
                    logger.warning(f"Failed to load Temporal Consistency checkpoint: {e}")
                    logger.info("Initializing Temporal Consistency with random weights...")
                    from .video_models import TemporalConsistencyModel
                    self.models['temporal_consistency'] = TemporalConsistencyModel()
                    self.models['temporal_consistency'].to(self.device)
                    self.models['temporal_consistency'].eval()
            elif (self.models_dir / "celebdf_config.json").exists():
                with open(self.models_dir / "celebdf_config.json", "r") as f:
                    config = json.load(f)
                
                self.models['temporal_consistency'] = TemporalConsistencyDetector(
                    num_classes=config.get('num_classes', 2)
                )
                self.models['temporal_consistency'].to(self.device)
                self.models['temporal_consistency'].eval()
                logger.info("Loaded Temporal Consistency model (config only)")
            
            if not self.models:
                logger.warning("No video models found, initializing default 3D CNN")
                self.models['3d_cnn'] = Video3DCNN()
                self.models['3d_cnn'].to(self.device)
                self.models['3d_cnn'].eval()
                
        except Exception as e:
            logger.error(f"Error loading video models: {e}")
    
    def detect_single_model(self, input_tensor: torch.Tensor, model_name: str) -> Dict:
        """Run detection with single model"""
        try:
            model = self.models[model_name]
            input_tensor = input_tensor.to(self.device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                
                # Get predictions
                avg_probs = torch.mean(probabilities, dim=0)
                fake_confidence = avg_probs[1].item() if len(avg_probs) > 1 else 0.0
                real_confidence = avg_probs[0].item()
                
            return {
                'model': model_name,
                'fake_confidence': fake_confidence,
                'real_confidence': real_confidence,
                'prediction': 'fake' if fake_confidence > real_confidence else 'real'
            }
            
        except Exception as e:
            logger.error(f"Error in {model_name} detection: {e}")
            return {
                'model': model_name,
                'error': str(e),
                'fake_confidence': 0.0,
                'real_confidence': 0.0,
                'prediction': 'unknown'
            }
    
    def frame_based_detection(self, frames: np.ndarray) -> Dict:
        """Run frame-based detection using image detector"""
        try:
            from .image_detector import ImageDeepfakeDetector
            image_detector = ImageDeepfakeDetector(device=self.device)
            
            frame_results = []
            
            # Sample frames for analysis
            num_frames_to_analyze = min(10, len(frames))
            frame_indices = np.linspace(0, len(frames)-1, num_frames_to_analyze, dtype=int)
            
            for frame_idx in frame_indices:
                frame = frames[frame_idx]
                
                # Create temporary file for frame with proper Windows handling
                tmp_file = None
                try:
                    # Create temp file
                    tmp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                    tmp_file_path = tmp_file.name
                    tmp_file.close()  # Close handle before PIL operations
                    
                    # Save frame to temp file
                    frame_pil = Image.fromarray((frame * 255).astype(np.uint8) if frame.dtype == np.float64 else frame.astype(np.uint8))
                    frame_pil.save(tmp_file_path)
                    frame_pil.close()  # Close PIL image
                    
                    # Detect on frame
                    result = image_detector.detect(tmp_file_path)
                    frame_results.append(result)
                    
                except Exception as frame_error:
                    logger.warning(f"Error processing frame {frame_idx}: {frame_error}")
                    frame_results.append({
                        'error': f'Frame processing failed: {frame_error}',
                        'fake_confidence': 0.0,
                        'real_confidence': 0.0,
                        'prediction': 'unknown'
                    })
                finally:
                    # Clean up temp file with proper Windows handling
                    if tmp_file:
                        try:
                            if os.path.exists(tmp_file_path):
                                os.unlink(tmp_file_path)
                        except OSError as cleanup_error:
                            logger.warning(f"Could not clean up temp file {tmp_file_path}: {cleanup_error}")
            
            # Aggregate frame results
            valid_results = [r for r in frame_results if 'error' not in r]
            if not valid_results:
                return {
                    'prediction': 'unknown',
                    'confidence': 0.0,
                    'fake_confidence': 0.0,
                    'real_confidence': 0.0,
                    'error': 'No valid frame detections'
                }
            
            avg_fake_conf = np.mean([r['fake_confidence'] for r in valid_results])
            avg_real_conf = np.mean([r['real_confidence'] for r in valid_results])
            
            return {
                'model': 'frame_based',
                'fake_confidence': float(avg_fake_conf),
                'real_confidence': float(avg_real_conf),
                'prediction': 'fake' if avg_fake_conf > avg_real_conf else 'real',
                'num_frames_analyzed': len(valid_results)
            }
            
        except Exception as e:
            logger.error(f"Error in frame-based detection: {e}")
            return {
                'model': 'frame_based',
                'error': str(e),
                'fake_confidence': 0.0,
                'real_confidence': 0.0,
                'prediction': 'unknown'
            }
    
    def ensemble_prediction(self, results: List[Dict]) -> Dict:
        """Combine predictions from multiple models"""
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            return {
                'prediction': 'unknown',
                'confidence': 0.0,
                'fake_confidence': 0.0,
                'real_confidence': 0.0,
                'models_used': [],
                'individual_results': results
            }
        
        # Weighted ensemble
        model_weights = {
            '3d_cnn': 0.4,
            'temporal_consistency': 0.4,
            'frame_based': 0.2
        }
        
        weighted_fake_conf = 0.0
        weighted_real_conf = 0.0
        total_weight = 0.0
        models_used = []
        
        for result in valid_results:
            model_name = result['model']
            weight = model_weights.get(model_name, 1.0)
            
            weighted_fake_conf += result['fake_confidence'] * weight
            weighted_real_conf += result['real_confidence'] * weight
            total_weight += weight
            models_used.append(model_name)
        
        if total_weight > 0:
            weighted_fake_conf /= total_weight
            weighted_real_conf /= total_weight
        
        # Apply prediction logic based on highest confidence score
        # FIX: Always choose the prediction with the highest confidence, not threshold-based
        if weighted_fake_conf > weighted_real_conf:
            prediction = 'fake'
            confidence = weighted_fake_conf
        else:
            prediction = 'real'
            confidence = weighted_real_conf
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'fake_confidence': weighted_fake_conf,
            'real_confidence': weighted_real_conf,
            'models_used': models_used,
            'individual_results': results
        }
    
    def detect(self, video_path: str) -> Dict:
        """Main detection method with improved error handling"""
        start_time = time.time()
        try:
            logger.info(f"🎬 Starting video detection for: {video_path}")
            
            # Extract frames
            frames = self.video_processor.extract_frames(video_path)
            if frames is None:
                logger.warning(f"Failed to extract frames from {video_path}")
                return {
                    'error': 'Failed to extract frames',
                    'prediction': 'unknown',
                    'confidence': 0.0,
                    'processing_time_ms': (time.time() - start_time) * 1000
                }
            
            # Extract face sequences
            face_sequences = self.video_processor.extract_face_sequences(frames)
            if not face_sequences:
                return {
                    'error': 'No faces detected in video',
                    'prediction': 'unknown',
                    'confidence': 0.0
                }
            
            results = []
            face_sequence = face_sequences[0]  # Use first/largest face sequence
            
            # 3D CNN analysis
            if '3d_cnn' in self.models:
                input_tensor = self.video_processor.preprocess_for_3d_cnn(face_sequence)
                if input_tensor is not None:
                    result = self.detect_single_model(input_tensor, '3d_cnn')
                    results.append(result)
            
            # Temporal consistency analysis
            if 'temporal_consistency' in self.models:
                input_tensor = self.video_processor.preprocess_for_temporal_consistency(face_sequence)
                if input_tensor is not None:
                    result = self.detect_single_model(input_tensor, 'temporal_consistency')
                    results.append(result)
            
            # Frame-based detection
            frame_result = self.frame_based_detection(frames)
            results.append(frame_result)
            
            # Ensemble prediction
            final_result = self.ensemble_prediction(results)
            
            # Add metadata and timing
            processing_time = (time.time() - start_time) * 1000
            final_result.update({
                'media_type': 'video',
                'num_frames_extracted': len(frames),
                'num_face_sequences': len(face_sequences),
                'video_duration_estimate': len(frames) / self.video_processor.target_fps,
                'device_used': str(self.device),
                'processing_time_ms': processing_time
            })
            
            logger.info(f"✅ Video detection completed in {processing_time:.1f}ms - Result: {final_result.get('prediction', 'unknown')}")
            return final_result
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"❌ Error in video detection after {processing_time:.1f}ms: {e}")
            return {
                'error': str(e),
                'prediction': 'unknown',
                'confidence': 0.0,
                'fake_confidence': 0.0,
                'real_confidence': 0.5,
                'media_type': 'video',
                'processing_time_ms': processing_time
            }
    
    def batch_detect(self, video_paths: List[str]) -> List[Dict]:
        """Batch detection for multiple videos"""
        results = []
        for video_path in video_paths:
            result = self.detect(video_path)
            result['file_path'] = video_path
            results.append(result)
        
        return results