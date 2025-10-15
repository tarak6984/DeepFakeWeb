import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleRawNet2(nn.Module):
    """Simplified RawNet2-style model for audio deepfake detection"""
    
    def __init__(self, num_classes=2):
        super().__init__()
        
        # Simplified architecture based on RawNet2
        self.first_conv = nn.Conv1d(1, 128, kernel_size=1024, stride=1, padding=512)
        self.first_bn = nn.BatchNorm1d(128)
        
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1000)
            ),
            nn.Sequential(
                nn.Conv1d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(500)
            ),
            nn.Sequential(
                nn.Conv1d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(250)
            )
        ])
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        
        x = torch.relu(self.first_bn(self.first_conv(x)))
        
        for block in self.conv_blocks:
            x = block(x)
        
        x = self.global_pool(x).squeeze(-1)
        x = self.classifier(x)
        
        return x

class SimpleAASIST(nn.Module):
    """Simplified AASIST-style model for audio deepfake detection"""
    
    def __init__(self, num_classes=2):
        super().__init__()
        
        # Spectro-temporal feature extraction
        self.spec_layer = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 1), padding=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        
        # Graph attention-like blocks (simplified)
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((16, 16))
            ),
            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((8, 8))
            )
        ])
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Compute spectrogram if input is raw audio
        if len(x.shape) == 2:
            # Convert to spectrogram
            x = torch.stft(x, n_fft=512, hop_length=256, win_length=512, 
                          window=torch.hann_window(512, device=x.device), 
                          return_complex=True)
            x = torch.abs(x).unsqueeze(1)  # Add channel dimension
        
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        # Spectro-temporal features
        x = self.spec_layer(x)
        
        # Graph attention blocks
        for block in self.conv_blocks:
            x = block(x)
        
        # Classification
        x = self.classifier(x)
        
        return x