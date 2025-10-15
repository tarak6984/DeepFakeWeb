#!/usr/bin/env python3
"""
Deepfake Detection Web App - Project Initialization Script
This script creates a complete deepfake detection project from scratch
"""

import os
import sys
import subprocess
import json
import shutil
from pathlib import Path
import requests
import tempfile
import zipfile
from datetime import datetime

class ProjectInitializer:
    def __init__(self, project_name="deepfake-detection-app"):
        self.project_name = project_name
        self.project_dir = Path.cwd() / project_name
        self.success_count = 0
        self.error_count = 0
        
    def log_success(self, message):
        print(f"✅ {message}")
        self.success_count += 1
        
    def log_error(self, message):
        print(f"❌ {message}")
        self.error_count += 1
        
    def log_info(self, message):
        print(f"ℹ️  {message}")
        
    def run_command(self, command, cwd=None, check=True):
        """Run shell command with proper error handling"""
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                cwd=cwd or self.project_dir,
                check=check,
                capture_output=True,
                text=True
            )
            return result
        except subprocess.CalledProcessError as e:
            self.log_error(f"Command failed: {command}")
            self.log_error(f"Error: {e.stderr}")
            return None

    def check_prerequisites(self):
        """Check if required tools are installed"""
        self.log_info("Checking prerequisites...")
        
        required_tools = {
            'node': 'Node.js',
            'npm': 'npm',
            'python': 'Python',
            'pip': 'pip',
            'git': 'Git'
        }
        
        missing_tools = []
        
        for command, name in required_tools.items():
            try:
                subprocess.run([command, '--version'], 
                             capture_output=True, check=True)
                self.log_success(f"{name} is installed")
            except (subprocess.CalledProcessError, FileNotFoundError):
                missing_tools.append(name)
                self.log_error(f"{name} is not installed")
        
        if missing_tools:
            self.log_error(f"Missing tools: {', '.join(missing_tools)}")
            self.log_info("Please install missing tools before continuing.")
            return False
            
        return True

    def create_project_structure(self):
        """Create basic project directory structure"""
        self.log_info("Creating project structure...")
        
        directories = [
            "",
            "src/app",
            "src/app/api/auth/[...nextauth]",
            "src/app/api/upload",
            "src/app/api/analyses",
            "src/app/auth/signin",
            "src/app/auth/signup",
            "src/app/dashboard",
            "src/app/upload",
            "src/app/results",
            "src/components/ui",
            "src/components/layout",
            "src/components/charts",
            "src/lib",
            "public",
            "prisma",
            "deepfake_api/detectors",
            "deepfake_api/models/audio",
            "deepfake_api/models/image",
            "deepfake_api/models/video",
            "deepfake_api/models/multimodal",
            "deepfake_api/uploads",
            "deepfake_api/temp",
            "deepfake_api/cache",
            "deepfake_api/logs",
        ]
        
        for dir_path in directories:
            full_path = self.project_dir / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            
        self.log_success("Project structure created")

    def initialize_nextjs_project(self):
        """Initialize Next.js project with TypeScript and Tailwind"""
        self.log_info("Initializing Next.js project...")
        
        # Create package.json
        package_json = {
            "name": self.project_name,
            "version": "0.1.0",
            "private": True,
            "scripts": {
                "dev": "next dev --turbo",
                "dev:fast": "SKIP_ENV_VALIDATION=true next dev --turbo --port 3000",
                "dev:api": "cd deepfake_api && python main.py",
                "dev:full": "concurrently \"npm run dev\" \"npm run dev:api\" --names \"NEXT,API\" --prefix-colors \"cyan,yellow\"",
                "build": "next build",
                "start": "next start",
                "lint": "eslint",
                "type-check": "tsc --noEmit"
            },
            "dependencies": {
                "@next-auth/prisma-adapter": "^1.0.7",
                "@prisma/client": "^6.17.1",
                "@radix-ui/react-avatar": "^1.1.10",
                "@radix-ui/react-checkbox": "^1.3.3",
                "@radix-ui/react-dialog": "^1.1.15",
                "@radix-ui/react-dropdown-menu": "^2.1.16",
                "@radix-ui/react-label": "^2.1.7",
                "@radix-ui/react-progress": "^1.1.7",
                "@radix-ui/react-select": "^2.2.6",
                "@radix-ui/react-separator": "^1.1.7",
                "@radix-ui/react-slot": "^1.2.3",
                "@radix-ui/react-switch": "^1.2.6",
                "@radix-ui/react-tabs": "^1.1.13",
                "@radix-ui/react-toast": "^1.2.15",
                "@radix-ui/react-toggle": "^1.1.10",
                "bcryptjs": "^3.0.2",
                "chart.js": "^4.5.0",
                "class-variance-authority": "^0.7.1",
                "clsx": "^2.1.1",
                "date-fns": "^4.1.0",
                "framer-motion": "^12.23.22",
                "html2canvas": "^1.4.1",
                "jspdf": "^3.0.3",
                "jsonwebtoken": "^9.0.2",
                "lucide-react": "^0.544.0",
                "next": "15.5.4",
                "next-auth": "^4.24.11",
                "next-themes": "^0.4.6",
                "nodemailer": "^6.10.1",
                "prisma": "^6.17.1",
                "react": "19.1.0",
                "react-dom": "19.1.0",
                "react-dropzone": "^14.3.8",
                "recharts": "^3.2.1",
                "sonner": "^2.0.7",
                "tailwind-merge": "^3.3.1"
            },
            "devDependencies": {
                "@types/bcryptjs": "^2.4.6",
                "@types/jsonwebtoken": "^9.0.10",
                "@types/nodemailer": "^7.0.2",
                "@types/node": "^20",
                "@types/react": "^19",
                "@types/react-dom": "^19",
                "autoprefixer": "^10.4.21",
                "concurrently": "^9.2.1",
                "eslint": "^9",
                "eslint-config-next": "15.5.4",
                "postcss": "^8.5.6",
                "tailwindcss": "^3.4.18",
                "typescript": "^5"
            }
        }
        
        with open(self.project_dir / "package.json", "w") as f:
            json.dump(package_json, f, indent=2)
            
        self.log_success("package.json created")

    def create_configuration_files(self):
        """Create configuration files"""
        self.log_info("Creating configuration files...")
        
        # TypeScript config
        tsconfig = {
            "compilerOptions": {
                "lib": ["dom", "dom.iterable", "es6"],
                "allowJs": True,
                "skipLibCheck": True,
                "strict": True,
                "noEmit": True,
                "esModuleInterop": True,
                "module": "esnext",
                "moduleResolution": "bundler",
                "resolveJsonModule": True,
                "isolatedModules": True,
                "jsx": "preserve",
                "incremental": True,
                "plugins": [{"name": "next"}],
                "baseUrl": ".",
                "paths": {"@/*": ["./src/*"]}
            },
            "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
            "exclude": ["node_modules"]
        }
        
        with open(self.project_dir / "tsconfig.json", "w") as f:
            json.dump(tsconfig, f, indent=2)
            
        # Tailwind config
        tailwind_config = """/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
"""
        
        with open(self.project_dir / "tailwind.config.js", "w") as f:
            f.write(tailwind_config)
            
        # Next.js config
        next_config = """/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    appDir: true,
  },
}

module.exports = nextConfig
"""
        
        with open(self.project_dir / "next.config.js", "w") as f:
            f.write(next_config)
            
        self.log_success("Configuration files created")

    def setup_prisma(self):
        """Setup Prisma database"""
        self.log_info("Setting up Prisma...")
        
        schema_content = """// This is your Prisma schema file,
// learn more about it in the docs: https://pris.ly/d/prisma-schema

generator client {
  provider = "prisma-client-js"
}

datasource db {
  provider = "sqlite"
  url      = env("DATABASE_URL")
}

model User {
  id            String    @id @default(cuid())
  name          String?
  email         String    @unique
  emailVerified DateTime?
  image         String?
  password      String?
  role          UserRole  @default(USER)
  createdAt     DateTime  @default(now())
  updatedAt     DateTime  @updatedAt

  accounts  Account[]
  sessions  Session[]
  analyses  Analysis[]

  @@map("users")
}

model Account {
  id                String  @id @default(cuid())
  userId            String
  type              String
  provider          String
  providerAccountId String
  refresh_token     String?
  access_token      String?
  expires_at        Int?
  token_type        String?
  scope             String?
  id_token          String?
  session_state     String?

  user User @relation(fields: [userId], references: [id], onDelete: Cascade)

  @@unique([provider, providerAccountId])
  @@map("accounts")
}

model Session {
  id           String   @id @default(cuid())
  sessionToken String   @unique
  userId       String
  expires      DateTime
  user         User     @relation(fields: [userId], references: [id], onDelete: Cascade)

  @@map("sessions")
}

model Analysis {
  id              String   @id @default(cuid())
  userId          String
  filename        String
  originalName    String
  fileType        String
  fileSize        Int
  confidence      Float
  prediction      Prediction
  processingTime  Int
  createdAt       DateTime @default(now())
  updatedAt       DateTime @updatedAt
  
  user            User     @relation(fields: [userId], references: [id], onDelete: Cascade)
  
  @@map("analyses")
}

enum UserRole {
  USER
  ADMIN
}

enum Prediction {
  AUTHENTIC
  MANIPULATED
  INCONCLUSIVE
}
"""
        
        with open(self.project_dir / "prisma" / "schema.prisma", "w") as f:
            f.write(schema_content)
            
        self.log_success("Prisma schema created")

    def create_python_backend(self):
        """Create Python FastAPI backend"""
        self.log_info("Creating Python backend...")
        
        # Requirements.txt
        requirements = """torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
transformers>=4.30.0
opencv-python>=4.8.0
Pillow>=9.5.0
librosa>=0.10.0
soundfile>=0.12.0
scipy>=1.10.0
numpy>=1.24.0
fastapi>=0.100.0
uvicorn>=0.23.0
python-multipart>=0.0.6
aiofiles>=23.1.0
python-dotenv>=1.0.0
pyyaml>=6.0
requests>=2.31.0
"""
        
        with open(self.project_dir / "deepfake_api" / "requirements.txt", "w") as f:
            f.write(requirements)
            
        # Main FastAPI app
        main_py = '''#!/usr/bin/env python3
"""
Deepfake Detection API
"""

import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import uvicorn

app = FastAPI(
    title="Deepfake Detection API",
    description="AI-powered deepfake detection for images, audio, and video",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Deepfake Detection API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/analyze")
async def analyze_file(file: UploadFile = File(...)):
    """Analyze uploaded file for deepfake detection"""
    # Placeholder implementation
    return {
        "filename": file.filename,
        "confidence": 0.95,
        "prediction": "authentic",
        "processing_time": 1234
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
'''
        
        with open(self.project_dir / "deepfake_api" / "main.py", "w") as f:
            f.write(main_py)
            
        # Create __init__.py files
        (self.project_dir / "deepfake_api" / "__init__.py").touch()
        (self.project_dir / "deepfake_api" / "detectors" / "__init__.py").touch()
        
        self.log_success("Python backend created")

    def create_environment_files(self):
        """Create environment configuration files"""
        self.log_info("Creating environment files...")
        
        env_content = '''# Database Configuration
DATABASE_URL="file:./prisma/dev.db"

# NextAuth.js Configuration
NEXTAUTH_URL=http://localhost:3000
NEXTAUTH_SECRET=your-secret-key-here-change-this-in-production

# Backend API URL
BACKEND_URL=http://localhost:8000
'''
        
        with open(self.project_dir / ".env", "w") as f:
            f.write(env_content)
            
        env_example = '''# Database Configuration
DATABASE_URL="file:./prisma/dev.db"

# NextAuth.js Configuration
NEXTAUTH_URL=http://localhost:3000
NEXTAUTH_SECRET=your-secret-key-here

# Email Configuration (optional)
EMAIL_SERVER_HOST=smtp.gmail.com
EMAIL_SERVER_PORT=587
EMAIL_SERVER_USER=your-email@gmail.com
EMAIL_SERVER_PASSWORD=your-app-password
EMAIL_FROM=your-email@gmail.com

# Backend API URL
BACKEND_URL=http://localhost:8000
'''
        
        with open(self.project_dir / ".env.example", "w") as f:
            f.write(env_example)
            
        self.log_success("Environment files created")

    def create_startup_scripts(self):
        """Create startup scripts"""
        self.log_info("Creating startup scripts...")
        
        # Bash startup script
        bash_script = '''#!/bin/bash
echo "🚀 Starting Deepfake Detection App..."
echo "===================================="

echo "📦 Installing dependencies..."
npm install

echo "🗄️ Setting up database..."
npx prisma generate
npx prisma db push

echo "🐍 Starting Python backend on port 8000..."
cd deepfake_api
python -m venv venv
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null
pip install -r requirements.txt
PYTHONPATH=. python main.py &
BACKEND_PID=$!
cd ..

echo "⏳ Waiting for backend to initialize..."
sleep 3

echo "⚛️  Starting Next.js frontend on port 3000..."
npm run dev &
FRONTEND_PID=$!

echo "✅ APPLICATION STARTED!"
echo "========================"
echo "🌐 Frontend: http://localhost:3000"
echo "🤖 Backend:  http://localhost:8000"
echo "📊 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all servers"

# Wait for user interrupt
trap 'echo "Stopping servers..."; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit' INT
wait
'''
        
        with open(self.project_dir / "start.sh", "w") as f:
            f.write(bash_script)
        
        # Make executable
        try:
            os.chmod(self.project_dir / "start.sh", 0o755)
        except:
            pass
            
        # Windows batch script
        batch_script = '''@echo off
echo 🚀 Starting Deepfake Detection App...
echo ====================================

echo 📦 Installing dependencies...
call npm install

echo 🗄️ Setting up database...
call npx prisma generate
call npx prisma db push

echo 🐍 Setting up Python environment...
cd deepfake_api
python -m venv venv
call venv\\Scripts\\activate.bat
pip install -r requirements.txt

echo 🚀 Starting backend...
start "Backend" cmd /k "set PYTHONPATH=. && python main.py"

cd ..
timeout /t 3 /nobreak >nul

echo ⚛️  Starting frontend...
start "Frontend" cmd /k "npm run dev"

echo ✅ APPLICATION STARTED!
echo ========================
echo 🌐 Frontend: http://localhost:3000
echo 🤖 Backend:  http://localhost:8000
echo 📊 API Docs: http://localhost:8000/docs

pause
'''
        
        with open(self.project_dir / "start.bat", "w") as f:
            f.write(batch_script)
            
        self.log_success("Startup scripts created")

    def create_readme(self):
        """Create comprehensive README"""
        self.log_info("Creating README...")
        
        readme_content = f'''# {self.project_name.replace('-', ' ').title()}

Advanced AI-powered deepfake detection platform built with Next.js and Python FastAPI.

## Features

- 🧠 **AI-Powered Detection**: Advanced machine learning models for deepfake detection
- 🖼️ **Multi-Media Support**: Images, videos, and audio analysis
- 👤 **User Management**: Authentication and user accounts
- 📊 **Analytics Dashboard**: Comprehensive analysis results and history
- 🔒 **Secure**: Built with security best practices
- ⚡ **Fast**: Optimized for performance

## Technology Stack

- **Frontend**: Next.js 15, React 19, TypeScript, Tailwind CSS
- **Backend**: Python FastAPI, AI/ML models
- **Database**: Prisma ORM with SQLite
- **Authentication**: NextAuth.js

## Quick Start

### Automated Setup
```bash
# Clone or download the project
# Navigate to project directory

# Linux/Mac
chmod +x start.sh
./start.sh

# Windows
start.bat
```

### Manual Setup
1. Install dependencies:
   ```bash
   npm install
   ```

2. Setup database:
   ```bash
   npx prisma generate
   npx prisma db push
   ```

3. Setup Python environment:
   ```bash
   cd deepfake_api
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\\Scripts\\activate.bat  # Windows
   pip install -r requirements.txt
   ```

4. Start the application:
   ```bash
   # Terminal 1: Backend
   cd deepfake_api
   PYTHONPATH=. python main.py

   # Terminal 2: Frontend
   npm run dev
   ```

## Access

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## Project Structure

```
{self.project_name}/
├── src/                    # Frontend source code
│   ├── app/               # Next.js app directory
│   ├── components/        # React components
│   └── lib/              # Utility functions
├── deepfake_api/          # Python backend
│   ├── detectors/         # AI detection modules
│   ├── models/           # AI model files
│   └── main.py           # FastAPI application
├── prisma/               # Database schema
├── public/               # Static assets
└── package.json          # Node.js dependencies
```

## Development

### Prerequisites
- Node.js 18+
- Python 3.9-3.11
- Git

### VS Code Setup
Recommended extensions:
- ES7+ React/Redux/React-Native snippets
- Python
- Prisma
- Tailwind CSS IntelliSense
- TypeScript Importer

### Available Scripts
- `npm run dev` - Start frontend development server
- `npm run build` - Build frontend for production
- `npm run lint` - Run ESLint
- `npm run type-check` - Run TypeScript type checking

## Deployment

See `PRODUCTION_DEPLOYMENT_GUIDE.md` for detailed production deployment instructions.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For support and questions, please check the documentation or create an issue.

---

**Created on {datetime.now().strftime('%Y-%m-%d')}**
'''
        
        with open(self.project_dir / "README.md", "w") as f:
            f.write(readme_content)
            
        self.log_success("README created")

    def install_dependencies(self):
        """Install Node.js dependencies"""
        self.log_info("Installing Node.js dependencies...")
        
        result = self.run_command("npm install")
        if result and result.returncode == 0:
            self.log_success("Node.js dependencies installed")
        else:
            self.log_error("Failed to install Node.js dependencies")

    def setup_database(self):
        """Setup Prisma database"""
        self.log_info("Setting up database...")
        
        # Generate Prisma client
        result = self.run_command("npx prisma generate")
        if result and result.returncode == 0:
            self.log_success("Prisma client generated")
        else:
            self.log_error("Failed to generate Prisma client")
            
        # Push database schema
        result = self.run_command("npx prisma db push")
        if result and result.returncode == 0:
            self.log_success("Database schema pushed")
        else:
            self.log_error("Failed to push database schema")

    def create_project(self, skip_install=False):
        """Main method to create the entire project"""
        print(f"🚀 Initializing {self.project_name}...")
        print("=" * 60)
        
        if not self.check_prerequisites():
            return False
            
        try:
            self.create_project_structure()
            self.initialize_nextjs_project()
            self.create_configuration_files()
            self.setup_prisma()
            self.create_python_backend()
            self.create_environment_files()
            self.create_startup_scripts()
            self.create_readme()
            
            if not skip_install:
                self.install_dependencies()
                self.setup_database()
            
            print("\n" + "=" * 60)
            print("🎉 PROJECT INITIALIZATION COMPLETE!")
            print(f"📊 Success: {self.success_count} | Errors: {self.error_count}")
            print(f"📂 Project created: {self.project_dir}")
            print("\n🚀 To start your application:")
            print(f"   cd {self.project_name}")
            print("   ./start.sh      (Linux/Mac)")
            print("   start.bat       (Windows)")
            print("\n🌐 Access URLs:")
            print("   Frontend: http://localhost:3000")
            print("   Backend:  http://localhost:8000")
            print("   API Docs: http://localhost:8000/docs")
            
            return True
            
        except Exception as e:
            self.log_error(f"Project initialization failed: {e}")
            return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize Deepfake Detection App")
    parser.add_argument("--name", default="deepfake-detection-app", help="Project name")
    parser.add_argument("--skip-install", action="store_true", help="Skip dependency installation")
    
    args = parser.parse_args()
    
    initializer = ProjectInitializer(args.name)
    success = initializer.create_project(args.skip_install)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()