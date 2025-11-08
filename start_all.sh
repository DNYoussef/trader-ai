#!/bin/bash

# ========================================
# UNIFIED STARTUP SCRIPT - Gary×Taleb Trading System
# Starts both backend and frontend servers with one command
# ========================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "  GARY×TALEB TRADING SYSTEM LAUNCHER"
echo "  Unified Dashboard + Trading Engine"
echo "========================================"
echo

# Function to check if a command exists
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}ERROR: $1 is not installed or not in PATH${NC}"
        echo "Please install $1 and try again"
        exit 1
    fi
}

# Function to check if a port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down services...${NC}"

    # Kill backend process
    if [ -f ".backend.pid" ]; then
        BACKEND_PID=$(cat .backend.pid)
        if kill -0 $BACKEND_PID 2>/dev/null; then
            kill $BACKEND_PID
            echo -e "${GREEN}✓ Backend server stopped${NC}"
        fi
        rm -f .backend.pid
    fi

    # Kill frontend process
    if [ -f ".frontend.pid" ]; then
        FRONTEND_PID=$(cat .frontend.pid)
        if kill -0 $FRONTEND_PID 2>/dev/null; then
            kill $FRONTEND_PID
            echo -e "${GREEN}✓ Frontend server stopped${NC}"
        fi
        rm -f .frontend.pid
    fi

    # Kill any remaining processes on ports
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    lsof -ti:3000 | xargs kill -9 2>/dev/null || true

    echo -e "\n${GREEN}All services stopped. Goodbye!${NC}\n"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM EXIT

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

check_command python3
check_command node
check_command npm

echo -e "${GREEN}✓ Prerequisites verified${NC}\n"

# Check if ports are already in use
if check_port 8000; then
    echo -e "${YELLOW}WARNING: Port 8000 is already in use${NC}"
    echo "Either the backend is already running or another service is using this port."
    read -p "Do you want to continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

if check_port 3000; then
    echo -e "${YELLOW}WARNING: Port 3000 is already in use${NC}"
    echo "Either the frontend is already running or another service is using this port."
    read -p "Do you want to continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

# ========================================
# BACKEND SERVER STARTUP
# ========================================
echo -e "${YELLOW}[1/3] Starting Backend Server (FastAPI + WebSocket)...${NC}"

# Activate virtual environment if it exists
if [ -f "venv/bin/activate" ]; then
    echo "Using virtual environment..."
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    echo "Using virtual environment..."
    source .venv/bin/activate
fi

# Install/update Python dependencies if needed
if [ ! -f ".backend_installed" ] || [ "requirements.txt" -nt ".backend_installed" ]; then
    echo "Installing Python dependencies..."
    pip install -r requirements.txt
    if [ $? -eq 0 ]; then
        touch .backend_installed
        echo -e "${GREEN}✓ Python dependencies installed${NC}"
    else
        echo -e "${RED}Failed to install Python dependencies${NC}"
        echo "Run manually: pip install -r requirements.txt"
    fi
else
    echo -e "${GREEN}✓ Python dependencies already installed${NC}"
fi

# Start backend server in background
echo "Launching backend server..."
cd src/dashboard
python3 run_server_simple.py > ../../logs/backend.log 2>&1 &
BACKEND_PID=$!
echo $BACKEND_PID > ../../.backend.pid
cd ../..

# Wait for backend to initialize
echo "Waiting for backend to start..."
for i in {1..30}; do
    if check_port 8000; then
        echo -e "${GREEN}✓ Backend server started on port 8000${NC}"
        break
    fi
    sleep 1
done

if ! check_port 8000; then
    echo -e "${RED}Failed to start backend server${NC}"
    exit 1
fi

# ========================================
# FRONTEND SERVER STARTUP
# ========================================
echo
echo -e "${YELLOW}[2/3] Starting Frontend Server (React + Vite)...${NC}"

# Check if node_modules exists and install if needed
cd src/dashboard/frontend
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies (this may take a few minutes)..."
    npm install
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to install frontend dependencies${NC}"
        echo "Run manually: cd src/dashboard/frontend && npm install"
        cd ../../..
        exit 1
    fi
    echo -e "${GREEN}✓ Frontend dependencies installed${NC}"
else
    echo -e "${GREEN}✓ Frontend dependencies already installed${NC}"
fi

# Start frontend in background
echo "Launching frontend server..."
npm run dev > ../../../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
echo $FRONTEND_PID > ../../../.frontend.pid
cd ../../..

# Wait for frontend to compile
echo "Waiting for frontend to compile..."
for i in {1..60}; do
    if check_port 3000; then
        echo -e "${GREEN}✓ Frontend server started on port 3000${NC}"
        break
    fi
    sleep 1
done

if ! check_port 3000; then
    echo -e "${RED}Frontend may still be compiling. Check http://localhost:3000 in a moment.${NC}"
fi

# ========================================
# FINAL STATUS
# ========================================
echo
echo -e "${YELLOW}[3/3] System Ready!${NC}"
sleep 2

echo
echo -e "${GREEN}========================================"
echo "   SYSTEM STARTED SUCCESSFULLY!"
echo "========================================${NC}"
echo
echo -e "${GREEN}📊 Frontend Dashboard:${NC} http://localhost:3000"
echo -e "${GREEN}📡 Backend API:${NC} http://localhost:8000"
echo -e "${GREEN}🔌 WebSocket:${NC} ws://localhost:8000/ws/{client_id}"
echo -e "${GREEN}📚 API Documentation:${NC} http://localhost:8000/docs"
echo
echo -e "${YELLOW}SERVICES RUNNING:${NC}"
echo "  • Backend Server (FastAPI) - PID: $BACKEND_PID"
echo "  • Frontend Server (Vite) - PID: $FRONTEND_PID"
echo
echo -e "${YELLOW}USEFUL COMMANDS:${NC}"
echo "  • View backend logs: tail -f logs/backend.log"
echo "  • View frontend logs: tail -f logs/frontend.log"
echo "  • Check API health: curl http://localhost:8000/api/health"
echo "  • Stop all services: Press Ctrl+C"
echo
echo -e "${YELLOW}TROUBLESHOOTING:${NC}"
echo "  • If frontend doesn't load, wait 10-15 seconds for compilation"
echo "  • Check logs for any error messages"
echo "  • Ensure no other applications are using ports 3000 or 8000"
echo

# Create logs directory if it doesn't exist
mkdir -p logs

# Open browser automatically (works on Mac and most Linux distros)
echo "Opening browser..."
sleep 2
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    open http://localhost:3000
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    xdg-open http://localhost:3000 2>/dev/null || echo "Please open http://localhost:3000 in your browser"
fi

echo
echo -e "${GREEN}System is running. Press Ctrl+C to stop all services...${NC}"

# Keep script running
while true; do
    sleep 1
done