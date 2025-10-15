# 🛡️ Local Deepfake Detection API

A powerful, cost-effective **local alternative** to Reality Defender API with support for **unlimited** deepfake detection on images, audio, and video files.

## 🌟 Features

- **Multi-Modal Detection**: Images, Audio, and Video support
- **State-of-the-Art Models**: EfficientNet-B7, Xception, RawNet2, AASIST, 3D CNN
- **Unlimited Usage**: No API limits or costs
- **High Accuracy**: Ensemble predictions from multiple specialized models
- **GPU Acceleration**: CUDA/MPS support for fast processing
- **Easy Integration**: Drop-in replacement for Reality Defender API
- **Batch Processing**: Analyze multiple files simultaneously
- **RESTful API**: FastAPI with automatic documentation

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional but recommended)
- At least 8GB RAM
- 10GB+ free disk space for models

### 2. Installation

```bash
# Clone or extract the API code
cd deepfake_api

# Run automatic setup
python setup.py
```

### 3. Start the API

```bash
# Method 1: Direct
python main.py

# Method 2: Using uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000

# Method 3: Development mode (with auto-reload)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Test the API

Open your browser and go to:
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/health
- **API Info**: http://localhost:8000/

## 📋 API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload` | Upload and analyze single file |
| `GET` | `/api/result/{id}` | Get analysis result |
| `POST` | `/api/batch` | Upload and analyze multiple files |
| `GET` | `/api/batch/{batch_id}` | Get batch results |
| `GET` | `/api/health` | Health check |

### Reality Defender Compatible

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/files/aws-presigned` | Compatible endpoint (returns local info) |
| `GET` | `/api/media/users/{id}` | Compatible result endpoint |

## 🔄 Replacing Reality Defender

### 1. Update Your Frontend Code

Replace your Reality Defender API calls:

```javascript
// OLD: Reality Defender
const response = await fetch('https://api.realitydefender.com/api/upload', {
  method: 'POST',
  headers: { 'X-API-KEY': 'your-key' },
  body: formData
});

// NEW: Local API
const response = await fetch('http://localhost:8000/api/upload', {
  method: 'POST',
  body: formData  // No API key needed!
});
```

### 2. Update Your Next.js Route Handlers

Update your API routes in `src/app/api/rd/`:

```typescript
// src/app/api/rd/signed-url/route.ts
export async function POST(req: Request) {
  // Replace Reality Defender with local API
  const { fileName } = await req.json();
  
  return NextResponse.json({
    message: "Local API - use /api/upload directly",
    upload_url: "http://localhost:8000/api/upload"
  });
}

// src/app/api/rd/result/[id]/route.ts
export async function GET(_req: Request, context: { params: Promise<{ id: string }> }) {
  const { id } = await context.params;
  
  const response = await fetch(`http://localhost:8000/api/result/${id}`);
  const data = await response.json();
  
  return NextResponse.json(data);
}
```

### 3. Update Environment Variables

```bash
# .env.local
NEXT_PUBLIC_RD_API_URL=http://localhost:8000
# Remove or comment out the API key - not needed!
# NEXT_PUBLIC_RD_API_KEY=your_api_key
```

## 🎯 Usage Examples

### Single File Analysis

```python
import requests

# Upload file
with open('suspect_image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/api/upload', files=files)
    result = response.json()
    analysis_id = result['analysis_id']

# Get result
result = requests.get(f'http://localhost:8000/api/result/{analysis_id}')
print(result.json())
```

### Batch Processing

```python
import requests

# Upload multiple files
files = [
    ('files', open('image1.jpg', 'rb')),
    ('files', open('audio1.wav', 'rb')),
    ('files', open('video1.mp4', 'rb'))
]

response = requests.post('http://localhost:8000/api/batch', files=files)
batch_info = response.json()

# Get batch results
batch_id = batch_info['batch_id']
results = requests.get(f'http://localhost:8000/api/batch/{batch_id}')
print(results.json())
```

### JavaScript/Fetch API

```javascript
async function analyzeFile(file) {
  const formData = new FormData();
  formData.append('file', file);
  
  // Upload
  const uploadResponse = await fetch('http://localhost:8000/api/upload', {
    method: 'POST',
    body: formData
  });
  
  const { analysis_id } = await uploadResponse.json();
  
  // Poll for results
  while (true) {
    const resultResponse = await fetch(`http://localhost:8000/api/result/${analysis_id}`);
    const result = await resultResponse.json();
    
    if (result.status === 'completed') {
      return result;
    } else if (result.status === 'failed') {
      throw new Error(result.error);
    }
    
    await new Promise(resolve => setTimeout(resolve, 1000)); // Wait 1 second
  }
}
```

## 🎛️ Configuration

Edit `config.yaml` to customize:

```yaml
api:
  host: "0.0.0.0"
  port: 8000
  max_file_size: 100  # MB
  cors_origins: ["http://localhost:3000"]

models:
  image:
    confidence_threshold: 0.5
  audio:
    confidence_threshold: 0.5
  video:
    confidence_threshold: 0.5

processing:
  gpu_enabled: true
  device: "auto"  # auto, cpu, cuda, mps
  batch_processing: true
```

## 📊 Model Information

### Image Detection Models
- **EfficientNet-B7**: Large-scale deepfake detection
- **Xception**: Face manipulation detection
- **Vision Transformer**: Attention-based analysis

### Audio Detection Models
- **RawNet2**: Raw waveform anti-spoofing
- **AASIST**: Spectro-temporal graph attention

### Video Detection Models
- **3D CNN**: Temporal inconsistency detection
- **Temporal Consistency**: LSTM + Attention
- **Frame-based**: Individual frame analysis

## 🔧 Advanced Setup

### Manual Model Download

```bash
# Download specific model repositories
python download_models.py
```

### GPU Setup

```bash
# Install CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or for macOS with MPS
pip install torch torchvision torchaudio
```

### Production Deployment

```bash
# Using gunicorn (recommended for production)
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Using Docker
docker build -t deepfake-api .
docker run -p 8000:8000 deepfake-api
```

## 📈 Performance Comparison

| Feature | Reality Defender | Local API |
|---------|------------------|-----------|
| **Cost** | $X per request | **FREE** |
| **Limits** | 50/month free | **Unlimited** |
| **Formats** | Image, Audio | **Image, Audio, Video** |
| **Speed** | Network dependent | **Local processing** |
| **Privacy** | Cloud processing | **Fully local** |
| **Accuracy** | Good | **Ensemble models** |
| **Customization** | None | **Full control** |

## 🐛 Troubleshooting

### Common Issues

1. **CUDA out of memory**
   ```yaml
   # config.yaml
   processing:
     device: "cpu"  # Force CPU usage
   ```

2. **Model download fails**
   ```bash
   # Manual download
   python download_models.py
   ```

3. **Port already in use**
   ```yaml
   # config.yaml
   api:
     port: 8001  # Use different port
   ```

### Logs and Debugging

```bash
# Enable debug mode
export DEEPFAKE_API_DEBUG=true
python main.py

# Check logs
tail -f deepfake_api.log
```

## 🤝 Integration with Your App

The API is designed as a **drop-in replacement** for Reality Defender. Most existing code will work with minimal changes:

1. Update the base URL
2. Remove API key requirements
3. Enjoy unlimited usage!

## 📝 Response Format

```json
{
  "analysis_id": "uuid-here",
  "prediction": "fake|real",
  "confidence": 0.85,
  "fake_confidence": 0.85,
  "real_confidence": 0.15,
  "media_type": "image|audio|video",
  "models_used": ["efficientnet_b7", "xception"],
  "individual_results": [...],
  "timestamp": "2024-01-01T12:00:00",
  "status": "completed"
}
```

## 📄 License

This project is open source and available under the MIT License.

---

🚀 **Enjoy unlimited, accurate deepfake detection without the costs!**