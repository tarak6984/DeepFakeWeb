# Deepfake Detection Project - Setup Guide
## Windows Git Bash Setup

## Prerequisites
Before starting, ensure you have these installed:
- **Node.js** (v18+): Download from [nodejs.org](https://nodejs.org/)
- **Python** (v3.8+): Download from [python.org](https://python.org/)
- **Git for Windows**: Download from [git-scm.com](https://git-scm.com/) (includes Git Bash)

## Quick Setup (5 minutes)

### 1. Extract and Navigate
```bash
# Extract the zip file (or use Windows Explorer to extract)
# Then open Git Bash in the extracted folder
# Right-click in the folder and select "Git Bash Here"
# OR navigate using cd command
cd /c/path/to/your/deepfakewebpythonapi
```

### 2. Install Node.js Dependencies
```bash
# Install all frontend dependencies
npm install
```

### 3. Install Python Dependencies
```bash
# Navigate to Python backend folder
cd deepfake_api

# Install Python dependencies
pip install -r requirements.txt

# Go back to project root
cd ..
```

### 4. Configure Environment (Optional)
```bash
# Copy the environment template if needed
cp .env.example .env

# Edit the .env file if you need to customize settings
# Use notepad or any text editor
notepad .env
# OR use vim if you prefer
vim .env
```

**Note:** The application uses local APIs, so no external API keys are required.

### 5. Run the Application

**Option A: One Command (Recommended)**
```bash
# In Git Bash, run the startup script
bash start-fullstack.sh
# OR
./start-fullstack.sh
```

**Option B: Manual Start**
```bash
# Git Bash Terminal 1: Start Python backend
cd deepfake_api
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Git Bash Terminal 2: Start Node.js frontend (open new Git Bash window)
npm run dev
```

## Access the Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Admin Panel**: Click "System" button in footer, password: `101010`

## Troubleshooting

### Common Issues:

**1. Port Already in Use**
```bash
# Kill processes on specific ports (Git Bash)
npx kill-port 3000 8000

# Or use Windows commands in Git Bash
netstat -ano | grep :3000
netstat -ano | grep :8000
# Then kill using taskkill /PID <PID> /F
```

**2. Python Dependencies Failed**
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install dependencies (use python, not python3 on Windows)
python -m pip install -r deepfake_api/requirements.txt

# Or create virtual environment
python -m venv venv
source venv/Scripts/activate  # Git Bash on Windows
pip install -r deepfake_api/requirements.txt
```

**3. Node.js Issues**
```bash
# Clear npm cache and reinstall
rm -rf node_modules package-lock.json
npm cache clean --force
npm install
```

**4. CUDA/GPU Issues**
```bash
# Install CPU-only version if GPU issues occur
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Development Commands

### Frontend (Next.js)
```bash
npm run dev          # Development server
npm run build        # Production build
npm run start        # Production server
npm run lint         # Check code quality
npm run type-check   # TypeScript validation
```

### Backend (Python)
```bash
cd deepfake_api
python main.py       # Start API server
```

### Full Stack
```bash
npm run dev:full     # Start both frontend and backend
npm run start:full   # Start both in production mode
```

## Project Structure
```
deepfakewebpythonapi/
├── src/                    # Next.js frontend source
├── deepfake_api/           # Python backend
│   ├── main.py            # FastAPI application
│   └── requirements.txt   # Python dependencies
├── public/                # Static assets
├── prisma/                # Database schema
├── package.json           # Node.js dependencies
├── .env                   # Environment variables
└── start-fullstack.sh     # Startup script
```

## System Requirements
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 5GB free space
- **OS**: Windows 10+, macOS 10.15+, or Linux
- **Internet**: Required for initial dependency installation and model downloads

## Support
If you encounter issues:
1. Check the console output for error messages
2. Ensure all dependencies are correctly installed
3. Check port availability (3000, 8000)
4. Verify both frontend and backend servers are running

## Stop the Application
```bash
# If using start-fullstack.sh in Git Bash
Ctrl+C

# Or kill specific processes
npx kill-port 3000 8000

# Or use Windows Task Manager to end Node.js and Python processes
```

---
**Note**: First run may take longer due to model downloads and dependency installation.