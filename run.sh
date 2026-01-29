#!/bin/bash

# Violence Detection System - Quick Start Script

set -e  # Exit on error

echo "🚨 Violence Detection System"
echo "=============================="
echo ""

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "❌ Error: .env file not found!"
    echo ""
    echo "Please create .env file from template:"
    echo "  cp .env.example .env"
    echo ""
    echo "Then edit .env and set your GEMINI_API_KEY and VIDEO_PATH"
    exit 1
fi

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 not found!"
    exit 1
fi

echo "✅ Configuration found"
echo "✅ Python 3 detected: $(python3 --version)"
echo ""

# Check if video file exists (read from .env)
VIDEO_PATH=$(grep "^VIDEO_PATH=" .env | cut -d '=' -f2)
if [ ! -f "$VIDEO_PATH" ]; then
    echo "⚠️  Warning: Video file not found at: $VIDEO_PATH"
    echo "   Please ensure the video file exists before running."
    echo ""
fi

echo "🚀 Starting Violence Detection System..."
echo ""

# Run the main script
python3 main.py
