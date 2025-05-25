#!/bin/bash
set -e

echo "🚀 Complete setup for Iranian Actor Image Crawler..."

# Update system
sudo apt update

# Install all required system dependencies
echo "📦 Installing system dependencies..."
sudo apt install -y \
    python3-full \
    python3-venv \
    python3-dev \
    build-essential \
    cmake \
    pkg-config \
    libx11-dev \
    libatlas-base-dev \
    libgtk-3-dev \
    libboost-python-dev \
    libopenblas-dev \
    liblapack-dev \
    libhdf5-dev \
    libboost-all-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    python3-opencv \
    gfortran \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev

# Install Google Chrome
echo "📦 Installing Google Chrome..."
wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | sudo apt-key add - 2>/dev/null || true
echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" | sudo tee /etc/apt/sources.list.d/google-chrome.list >/dev/null
sudo apt update
sudo apt install google-chrome-stable -y

# Install ChromeDriver
echo "📦 Installing ChromeDriver..."
sudo apt install chromium-chromedriver -y

# Verify Chrome installation
echo "✅ Verifying Chrome installation..."
google-chrome --version
chromedriver --version

# Create virtual environment
if [ ! -d "iranian_actors_env" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv iranian_actors_env
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source iranian_actors_env/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install Python packages in order
echo "📦 Installing Python packages..."

# Install numpy first
echo "Installing numpy..."
pip install numpy

# Install basic packages
echo "Installing basic packages..."
pip install requests beautifulsoup4 pillow selenium

# Install OpenCV
echo "Installing OpenCV..."
pip install opencv-python

# Install dlib (this takes time)
echo "🔧 Installing dlib (this may take 10-15 minutes)..."
pip install --no-cache-dir dlib

# Install face-recognition
echo "🔧 Installing face-recognition..."
pip install face-recognition

# Test the installation
echo "🧪 Testing installation..."
python3 -c "
import cv2
import face_recognition
import selenium
import requests
from bs4 import BeautifulSoup
from PIL import Image
import numpy as np
print('✅ All packages imported successfully!')
"

echo "✅ Installation complete!"
echo ""
echo "To run the crawler:"
echo "source iranian_actors_env/bin/activate"
echo "python iranian_actor_crawler_complete.py"