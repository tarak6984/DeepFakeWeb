#!/bin/bash

echo "===================================="
echo "🚀 STARTING FULLSTACK APPLICATION"
echo "===================================="
echo

echo "📦 Installing dependencies..."
npm install

echo
echo "🐍 Starting Python AI Backend on port 8000..."
cd deepfake_api
PYTHONPATH=. python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
cd ..

echo
echo "⏳ Waiting for backend to initialize..."
sleep 5

echo
echo "⚛️  Starting Next.js Frontend on port 3000..."
npm run dev &
FRONTEND_PID=$!

echo
echo "✅ FULLSTACK APPLICATION STARTED!"
echo "===================================="
echo "🌐 Frontend: http://localhost:3000"
echo "🤖 Backend:  http://localhost:8000"
echo "📊 API Docs: http://localhost:8000/docs"
echo "===================================="
echo
echo "🔐 Admin Access: Scroll to footer and click 'System' button"
echo "🔑 Password: 101010"
echo
echo "Press Ctrl+C to stop all servers"

# Wait for user interrupt
trap 'echo "Stopping servers..."; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit' INT
wait