import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalConsistencyModel(nn.Module):
    """Temporal consistency model for video deepfake detection"""
    
    def __init__(self, feature_dim=512, num_classes=2):
        super().__init__()
        
        # Feature extractor (simplified ResNet-like)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, 
                           num_layers=2, batch_first=True, dropout=0.3)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, channels, height, width)
        batch_size, seq_len, c, h, w = x.shape
        
        # Extract features for each frame
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.feature_extractor(x)  # (batch_size * seq_len, feature_dim)
        features = features.view(batch_size, seq_len, -1)  # (batch_size, seq_len, feature_dim)
        
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(features)
        
        # Use last output
        temporal_features = lstm_out[:, -1, :]
        
        # Classification
        output = self.classifier(temporal_features)
        
        return output