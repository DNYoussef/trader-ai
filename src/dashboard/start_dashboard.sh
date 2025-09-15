#!/bin/bash

# Gary×Taleb Risk Dashboard Startup Script
# This script starts both the WebSocket server and frontend development server

set -e

echo "🚀 Starting Gary×Taleb Risk Dashboard..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if a port is in use
check_port() {
    local port=$1
    if netstat -tuln 2>/dev/null | grep -q ":$port "; then
        return 0
    else
        return 1
    fi
}

# Function to start backend server
start_backend() {
    echo -e "${YELLOW}📡 Starting WebSocket server...${NC}"

    # Check if port 8000 is already in use
    if check_port 8000; then
        echo -e "${RED}❌ Port 8000 is already in use. Please stop the existing service or use a different port.${NC}"
        exit 1
    fi

    # Create logs directory if it doesn't exist
    mkdir -p logs

    # Install Python dependencies if requirements.txt is newer than the last install
    if [ "server/requirements.txt" -nt ".backend_deps_installed" ] || [ ! -f ".backend_deps_installed" ]; then
        echo -e "${YELLOW}📦 Installing/updating Python dependencies...${NC}"
        pip install -r server/requirements.txt
        touch .backend_deps_installed
    fi

    # Start the server in background
    python run_server.py > logs/server.log 2>&1 &
    BACKEND_PID=$!
    echo $BACKEND_PID > .backend.pid

    # Wait for server to start
    echo -e "${YELLOW}⏳ Waiting for server to start...${NC}"
    for i in {1..30}; do
        if check_port 8000; then
            echo -e "${GREEN}✅ WebSocket server started on http://localhost:8000${NC}"
            return 0
        fi
        sleep 1
    done

    echo -e "${RED}❌ Failed to start WebSocket server${NC}"
    return 1
}

# Function to start frontend
start_frontend() {
    echo -e "${YELLOW}🎨 Starting frontend development server...${NC}"

    cd frontend

    # Check if port 3000 is already in use
    if check_port 3000; then
        echo -e "${RED}❌ Port 3000 is already in use. Please stop the existing service or use a different port.${NC}"
        cd ..
        exit 1
    fi

    # Install Node dependencies if package.json is newer than the last install
    if [ "package.json" -nt ".frontend_deps_installed" ] || [ ! -d "node_modules" ]; then
        echo -e "${YELLOW}📦 Installing/updating Node.js dependencies...${NC}"
        npm install
        touch .frontend_deps_installed
    fi

    # Start development server
    echo -e "${YELLOW}⏳ Starting Vite development server...${NC}"
    npm run dev &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > ../.frontend.pid

    cd ..

    # Wait for frontend to start
    for i in {1..30}; do
        if check_port 3000; then
            echo -e "${GREEN}✅ Frontend development server started on http://localhost:3000${NC}"
            return 0
        fi
        sleep 1
    done

    echo -e "${RED}❌ Failed to start frontend development server${NC}"
    return 1
}

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}🛑 Shutting down dashboard...${NC}"

    if [ -f ".backend.pid" ]; then
        BACKEND_PID=$(cat .backend.pid)
        kill $BACKEND_PID 2>/dev/null || true
        rm .backend.pid
        echo -e "${GREEN}✅ WebSocket server stopped${NC}"
    fi

    if [ -f ".frontend.pid" ]; then
        FRONTEND_PID=$(cat .frontend.pid)
        kill $FRONTEND_PID 2>/dev/null || true
        rm .frontend.pid
        echo -e "${GREEN}✅ Frontend server stopped${NC}"
    fi

    echo -e "${GREEN}👋 Dashboard shutdown complete${NC}"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Check prerequisites
echo -e "${YELLOW}🔍 Checking prerequisites...${NC}"

# Check Python
if ! command -v python &> /dev/null; then
    echo -e "${RED}❌ Python is not installed or not in PATH${NC}"
    exit 1
fi

# Check Node.js
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js is not installed or not in PATH${NC}"
    exit 1
fi

# Check npm
if ! command -v npm &> /dev/null; then
    echo -e "${RED}❌ npm is not installed or not in PATH${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites check passed${NC}"

# Start services
start_backend
start_frontend

echo -e "\n${GREEN}🎉 Dashboard is ready!${NC}"
echo -e "${GREEN}📊 Frontend: http://localhost:3000${NC}"
echo -e "${GREEN}📡 Backend API: http://localhost:8000${NC}"
echo -e "${GREEN}🔌 WebSocket: ws://localhost:8000/ws${NC}"
echo -e "\n${YELLOW}📋 Useful commands:${NC}"
echo -e "  • View server logs: tail -f logs/server.log"
echo -e "  • Health check: curl http://localhost:8000/api/health"
echo -e "  • Stop dashboard: Ctrl+C"

# Keep script running
echo -e "\n${YELLOW}🔄 Dashboard is running. Press Ctrl+C to stop.${NC}"
while true; do
    sleep 1
done