#!/bin/bash

# Exercise Analyzer - Linux Setup Script
# For Linux Mint / Ubuntu / Debian

echo "🏋️ Exercise Analyzer Setup for Linux"
echo "======================================"

# Check Python version
python3 --version || { echo "❌ Python 3 not found. Installing..."; sudo apt update && sudo apt install -y python3 python3-pip python3-venv; }

# Install system dependencies for OpenCV and MediaPipe
echo ""
echo "📦 Installing system dependencies..."
sudo apt update
sudo apt install -y \
    python3-dev \
    python3-pip \
    python3-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libfontconfig1 \
    libice6 \
    ffmpeg

# Create virtual environment
echo ""
echo "🐍 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo ""
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install Python packages
echo ""
echo "📚 Installing Python packages (this may take a few minutes)..."
pip install -r requirements.txt

# Run migrations
echo ""
echo "🗄️ Setting up database..."
python manage.py migrate

# Create uploads directory
mkdir -p uploaded_videos

echo ""
echo "✅ Setup complete!"
echo ""
echo "To run the app:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Start the server: python manage.py runserver 0.0.0.0:8000"
echo ""
echo "To share with friends using ngrok:"
echo "  1. Install ngrok: https://ngrok.com/download"
echo "  2. Run: ngrok http 8000"
echo "  3. Share the https URL with friends!"
