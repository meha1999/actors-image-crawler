#!/bin/bash
set -e

echo "🚀 Complete setup for Iranian Actor Image Crawler..."

# Update system packages
sudo apt update

# Install system dependencies
echo "📦 Installing system dependencies..."
sudo apt install -y \
    python3-full \
    python3-venv \
    python3-dev \
    python3-pip \
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

# Remove existing virtual environment if it exists
if [ -d "iranian_actors_env" ]; then
    echo "🗑️ Removing existing virtual environment..."
    rm -rf iranian_actors_env
fi

# Create fresh virtual environment
echo "🔧 Creating virtual environment..."
python3 -m venv iranian_actors_env

# Define the virtual environment paths
VENV_PYTHON="./iranian_actors_env/bin/python"
VENV_PIP="./iranian_actors_env/bin/pip"

# Verify virtual environment was created properly
echo "🔍 Verifying virtual environment..."
echo "Python path: $($VENV_PYTHON -c 'import sys; print(sys.executable)')"
echo "Pip path: $(which $VENV_PIP 2>/dev/null || echo 'Using relative path')"

# Upgrade pip in virtual environment
echo "⬆️ Upgrading pip in virtual environment..."
$VENV_PIP install --upgrade pip setuptools wheel

# Install Python packages
echo "📦 Installing Python packages..."

echo "Installing numpy..."
$VENV_PIP install numpy

echo "Installing basic packages..."
$VENV_PIP install requests beautifulsoup4 pillow selenium

echo "Installing OpenCV..."
$VENV_PIP install opencv-python

echo "🔧 Installing dlib (this may take 10-15 minutes)..."
$VENV_PIP install --no-cache-dir dlib

echo "🔧 Installing face-recognition..."
$VENV_PIP install face-recognition

# Test installation
echo "🧪 Testing installation..."
$VENV_PYTHON -c "
import cv2
import face_recognition
import selenium
import requests
from bs4 import BeautifulSoup
from PIL import Image
import numpy as np
print('✅ All packages imported successfully!')
print('Python executable:', __import__('sys').executable)
"

echo "✅ Installation complete!"
echo ""
echo "To run the crawler:"
echo "source iranian_actors_env/bin/activate"
echo "python iranian_actor_crawler_complete.py"
echo ""
echo "Or run directly with:"
echo "./iranian_actors_env/bin/python iranian_actor_crawler_complete.py"