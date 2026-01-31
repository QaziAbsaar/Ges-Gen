#!/usr/bin/env python
"""
Gesture+Gen - Main Entry Point
==============================
Run the gesture-based drawing application.
"""

import sys
import os
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.ui import main

if __name__ == "__main__":
    main()
