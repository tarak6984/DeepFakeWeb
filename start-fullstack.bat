@echo off
echo ====================================
echo 🚀 STARTING FULLSTACK APPLICATION
echo ====================================

echo.
echo 📦 Installing dependencies...
call npm install

echo.
echo 🐍 Starting Python AI Backend on port 8000...
start "Python Backend" cmd /k "cd deepfake_api && python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

echo.
echo ⏳ Waiting for backend to initialize...
timeout /t 5 /nobreak > nul

echo.
echo ⚛️  Starting Next.js Frontend on port 3000...
start "Next.js Frontend" cmd /k "npm run dev"

echo.
echo ✅ FULLSTACK APPLICATION STARTED!
echo ====================================
echo 🌐 Frontend: http://localhost:3000
echo 🤖 Backend:  http://localhost:8000
echo 📊 API Docs: http://localhost:8000/docs
echo ====================================
echo.
echo 🔐 Admin Access: Scroll to footer and click "System" button
echo 🔑 Password: 101010
echo.
echo Press any key to exit...
pause > nul