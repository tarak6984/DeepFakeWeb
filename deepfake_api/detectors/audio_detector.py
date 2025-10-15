import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
import soundfile as sf
from scipy import signal
import logging
import yaml
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from pydub import AudioSegment

logger = logging.getLogger(__name__)

class RawNet2(nn.Module):
    """RawNet2 architecture for audio anti-spoofing"""
    
    def __init__(self, nb_samp=64600, first_conv=1024, in_channels=1, 
                 filts=[20, [20, 20], [20, 128], [128, 128]], 
                 gat_dims=[64, 32], pool_ratios=[0.5, 0.7, 0.5, 0.5], 
                 temperatures=[2.0, 2.0, 100.0, 100.0]):
        super().__init__()
        
        self.nb_samp = nb_samp
        self.first_conv = first_conv
        
        # First convolution
        self.first_bn = nn.BatchNorm1d(num_features=1)
        self.sincconv = nn.Conv1d(in_channels=in_channels, 
                                  out_channels=filts[0], 
                                  kernel_size=first_conv, 
                                  stride=1, 
                                  padding=first_conv//2, 
                                  bias=False)
        
        # Residual blocks
        self.block0 = self._make_layer(nb_filts=filts[1], 
                                      pool_ratio=pool_ratios[0], 
                                      temperature=temperatures[0])
        self.block1 = self._make_layer(nb_filts=filts[2], 
                                      pool_ratio=pool_ratios[1], 
                                      temperature=temperatures[1])
        self.block2 = self._make_layer(nb_filts=filts[3], 
                                      pool_ratio=pool_ratios[2], 
                                      temperature=temperatures[2])
        
        # GAT layers
        self.gat = nn.MultiheadAttention(embed_dim=gat_dims[0], num_heads=gat_dims[1])
        
        # Output layers
        self.fc_attention = self._make_attention_fc(in_dim=filts[3][1], gat_dim=gat_dims[0])
        self.fc = nn.Linear(gat_dims[0], 2)
        
    def _make_layer(self, nb_filts, pool_ratio, temperature):
        """Create residual block"""
        layers = []
        layers.append(nn.Conv1d(in_channels=nb_filts[0], 
                               out_channels=nb_filts[1], 
                               kernel_size=3, 
                               padding=1))
        layers.append(nn.BatchNorm1d(nb_filts[1]))
        layers.append(nn.ReLU())
        layers.append(nn.AdaptiveAvgPool1d(output_size=int(nb_filts[1] * pool_ratio)))
        return nn.Sequential(*layers)
    
    def _make_attention_fc(self, in_dim, gat_dim):
        """Create attention fully connected layer"""
        return nn.Sequential(
            nn.Linear(in_dim, gat_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        
        # First convolution
        x = self.first_bn(x)
        x = torch.abs(self.sincconv(x))
        
        # Residual blocks
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        
        # Global average pooling
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        
        # Attention
        x = self.fc_attention(x).unsqueeze(0)  # Add sequence dimension
        x, _ = self.gat(x, x, x)
        x = x.squeeze(0)  # Remove sequence dimension
        
        # Final classification
        x = self.fc(x)
        
        return x

class AASIST(nn.Module):
    """AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal graph attention)"""
    
    def __init__(self, nb_samp=64600, first_conv=251, 
                 filts=[70, [1, 32], [32, 32], [32, 64], [64, 64]],
                 gat_dims=[64, 32], pool_ratios=[0.5, 0.7, 0.5, 0.5],
                 temperatures=[2.0, 2.0, 100.0, 100.0]):
        super().__init__()
        
        self.nb_samp = nb_samp
        
        # Spectro-temporal feature extraction
        self.spec_layer = nn.Sequential(
            nn.Conv2d(1, filts[0], kernel_size=(7, 7), stride=(2, 1), padding=(3, 3)),
            nn.BatchNorm2d(filts[0]),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        
        # Graph attention blocks
        self.conv_blocks = nn.ModuleList()
        for i in range(len(filts) - 1):
            if i == 0:
                in_ch, out_ch = filts[0], filts[1][1]
            else:
                in_ch, out_ch = filts[i][1], filts[i+1][1]
            
            self.conv_blocks.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((int(32 * pool_ratios[i]), int(32 * pool_ratios[i])))
            ))
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(filts[-1][1], gat_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(gat_dims[0], 2)
        )
    
    def forward(self, x):
        # Compute spectrogram
        x = torch.stft(x, n_fft=512, hop_length=256, win_length=512, 
                      window=torch.hann_window(512, device=x.device), 
                      return_complex=True)
        x = torch.abs(x).unsqueeze(1)  # Add channel dimension
        
        # Spectro-temporal features
        x = self.spec_layer(x)
        
        # Graph attention blocks
        for block in self.conv_blocks:
            x = block(x)
        
        # Classification
        x = self.classifier(x)
        
        return x

class AudioPreprocessor:
    """Audio preprocessing utilities"""
    
    def __init__(self, sample_rate: int = 16000, duration: float = 3.0, fast_mode: bool = True, ultra_fast_mode: bool = True):
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_length = int(sample_rate * duration)
        self.fast_mode = fast_mode
        self.ultra_fast_mode = ultra_fast_mode
        self.max_segments = 5 if ultra_fast_mode else 10
    
    def load_audio(self, audio_path: str) -> Optional[np.ndarray]:
        """Load and preprocess audio file with optimization for large files"""
        try:
            # For ultra fast mode, limit duration to speed up large files
            max_duration = None
            if self.ultra_fast_mode:
                max_duration = min(30, self.duration * 10)  # Max 30 seconds for ultra fast
            elif self.fast_mode:
                max_duration = min(60, self.duration * 15)  # Max 60 seconds for fast
            
            # Try loading with librosa first (with duration limit)
            try:
                audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=max_duration)
            except:
                # Fallback to pydub for various formats
                audio_segment = AudioSegment.from_file(audio_path)
                
                # Limit duration for large files
                if max_duration and len(audio_segment) > max_duration * 1000:
                    audio_segment = audio_segment[:int(max_duration * 1000)]
                
                audio_segment = audio_segment.set_frame_rate(self.sample_rate).set_channels(1)
                audio = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                
                # Normalize only if needed
                max_val = np.max(np.abs(audio))
                if max_val > 0:
                    audio = audio / max_val
                sr = self.sample_rate
            
            # Skip resampling if already correct (saves time)
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            
            return audio
            
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            return None
    
    def segment_audio(self, audio: np.ndarray, overlap: float = 0.5) -> List[np.ndarray]:
        """Segment audio into fixed-length chunks"""
        if len(audio) <= self.target_length:
            # Pad if too short
            padded = np.zeros(self.target_length)
            padded[:len(audio)] = audio
            return [padded]
        
        # Optimize for different speed modes
        if self.ultra_fast_mode:
            overlap = 0.1  # Minimal overlap for ultra speed
            max_segments = min(2, self.max_segments)  # Ultra fast: max 2 segments
        elif self.fast_mode:
            overlap = 0.25  # Reduced overlap for speed
            max_segments = min(3, self.max_segments)  # Fast: max 3 segments
        else:
            overlap = 0.5  # Standard overlap
            max_segments = self.max_segments
        
        # Create overlapping segments
        step_size = int(self.target_length * (1 - overlap))
        segments = []
        
        for start in range(0, len(audio) - self.target_length + 1, step_size):
            if len(segments) >= max_segments:
                break
            segment = audio[start:start + self.target_length]
            segments.append(segment)
        
        # If no segments created (shouldn't happen), create one from the beginning
        if not segments:
            segments = [audio[:self.target_length]]
        
        return segments
    
    def apply_vad(self, audio: np.ndarray, frame_length: int = 2048, 
                  hop_length: int = 512) -> np.ndarray:
        """Apply Voice Activity Detection to remove silence"""
        try:
            # Simple energy-based VAD
            frames = librosa.util.frame(audio, frame_length=frame_length, 
                                       hop_length=hop_length, axis=0)
            energy = np.sum(frames ** 2, axis=0)
            
            # Threshold based on percentile
            threshold = np.percentile(energy, 30)
            voiced_frames = energy > threshold
            
            # Reconstruct audio with voiced frames only
            voiced_audio = []
            for i, is_voiced in enumerate(voiced_frames):
                if is_voiced:
                    start = i * hop_length
                    end = start + hop_length
                    voiced_audio.extend(audio[start:min(end, len(audio))])
            
            return np.array(voiced_audio) if voiced_audio else audio
            
        except:
            # Return original audio if VAD fails
            return audio

class AudioDeepfakeDetector:
    """Advanced audio deepfake detection system"""
    
    def __init__(self, models_dir: str = "models/audio", device: str = "auto"):
        self.models_dir = Path(models_dir)
        self.device = self._setup_device(device)
        self.preprocessor = AudioPreprocessor()
        
        # Load config for thresholds
        self.config = self._load_config()
        self.confidence_threshold = self.config.get('models', {}).get('audio', {}).get('confidence_threshold', 0.5)
        
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
        """Load pre-trained audio deepfake detection models"""
        try:
            # Load RawNet2 model
            if (self.models_dir / "rawnet2_model.pth").exists():
                # Load the actual model weights
                from .audio_models import SimpleRawNet2
                self.models['rawnet2'] = SimpleRawNet2()
                state_dict = torch.load(
                    self.models_dir / "rawnet2_model.pth", 
                    map_location=self.device,
                    weights_only=True
                )
                self.models['rawnet2'].load_state_dict(state_dict)
                self.models['rawnet2'].to(self.device)
                self.models['rawnet2'].eval()
                logger.info("Loaded RawNet2 model with trained weights")
            elif (self.models_dir / "rawnet2_config.json").exists():
                with open(self.models_dir / "rawnet2_config.json", "r") as f:
                    config = json.load(f)
                
                self.models['rawnet2'] = RawNet2(
                    nb_samp=config.get('input_features', 64600)
                )
                self.models['rawnet2'].to(self.device)
                self.models['rawnet2'].eval()
                logger.info("Loaded RawNet2 model (config only)")
            
            # Load AASIST model
            if (self.models_dir / "aasist_model.pth").exists():
                # Load the actual model weights
                from .audio_models import SimpleAASIST
                self.models['aasist'] = SimpleAASIST()
                state_dict = torch.load(
                    self.models_dir / "aasist_model.pth", 
                    map_location=self.device,
                    weights_only=True
                )
                self.models['aasist'].load_state_dict(state_dict)
                self.models['aasist'].to(self.device)
                self.models['aasist'].eval()
                logger.info("Loaded AASIST model with trained weights")
            elif (self.models_dir / "aasist_config.json").exists():
                with open(self.models_dir / "aasist_config.json", "r") as f:
                    config = json.load(f)
                
                self.models['aasist'] = AASIST(
                    nb_samp=config.get('nb_samp', 64600),
                    first_conv=config.get('first_conv', 251),
                    filts=config.get('filts', [70, [1, 32], [32, 32], [32, 64], [64, 64]]),
                    gat_dims=config.get('gat_dims', [64, 32]),
                    pool_ratios=config.get('pool_ratios', [0.5, 0.7, 0.5, 0.5]),
                    temperatures=config.get('temperatures', [2.0, 2.0, 100.0, 100.0])
                )
                self.models['aasist'].to(self.device)
                self.models['aasist'].eval()
                logger.info("Loaded AASIST model (config only)")
            
            if not self.models:
                logger.warning("No audio models found, initializing default RawNet2")
                self.models['rawnet2'] = RawNet2()
                self.models['rawnet2'].to(self.device)
                self.models['rawnet2'].eval()
                
        except Exception as e:
            logger.error(f"Error loading audio models: {e}")
    
    def detect_single_model(self, audio_tensor: torch.Tensor, model_name: str) -> Dict:
        """Run detection with single model"""
        try:
            model = self.models[model_name]
            audio_tensor = audio_tensor.to(self.device)
            
            with torch.no_grad():
                outputs = model(audio_tensor)
                probabilities = F.softmax(outputs, dim=1)
                
                # Average predictions across all segments
                avg_probs = torch.mean(probabilities, dim=0)
                fake_confidence = avg_probs[1].item()
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
            'rawnet2': 0.6,
            'aasist': 0.4
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
    
    def detect(self, audio_path: str) -> Dict:
        """Main detection method"""
        try:
            # Load and preprocess audio
            audio = self.preprocessor.load_audio(audio_path)
            if audio is None:
                return {
                    'error': 'Failed to load audio',
                    'prediction': 'unknown',
                    'confidence': 0.0
                }
            
            # Skip VAD in ultra-fast mode or fast mode for speed
            if not self.preprocessor.ultra_fast_mode and not self.preprocessor.fast_mode:
                audio = self.preprocessor.apply_vad(audio)
            
            # Segment audio
            segments = self.preprocessor.segment_audio(audio)
            audio_tensor = torch.FloatTensor(np.array(segments))
            
            # Run detection with all available models
            results = []
            for model_name in self.models.keys():
                result = self.detect_single_model(audio_tensor, model_name)
                results.append(result)
            
            # Ensemble prediction
            final_result = self.ensemble_prediction(results)
            
            # Add metadata
            final_result.update({
                'media_type': 'audio',
                'duration_seconds': len(audio) / self.preprocessor.sample_rate,
                'num_segments': len(segments),
                'sample_rate': self.preprocessor.sample_rate,
                'device_used': str(self.device)
            })
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error in audio detection: {e}")
            return {
                'error': str(e),
                'prediction': 'unknown',
                'confidence': 0.0,
                'media_type': 'audio'
            }
    
    def batch_detect(self, audio_paths: List[str]) -> List[Dict]:
        """Batch detection for multiple audio files"""
        results = []
        for audio_path in audio_paths:
            result = self.detect(audio_path)
            result['file_path'] = audio_path
            results.append(result)
        
        return results