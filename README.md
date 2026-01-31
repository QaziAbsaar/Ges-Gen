# Gesture+Gen 🎨✨

> Real-time gesture-based drawing system with AI image generation

Draw in the air using hand gestures captured through your webcam. Your sketches are then transformed into polished AI-generated images using Stable Diffusion.

![Hero Banner](images/Hero_Banner_image.png)

## 🌟 Features

- **Real-time Hand Tracking**: MediaPipe-powered finger and gesture detection
- **Air Drawing**: Use your index finger to draw on a virtual canvas
- **Gesture Controls**: Intuitive hand gestures for all actions
- **AI Image Generation**: Transform sketches into artwork using Stable Diffusion
- **Multiple Styles**: Choose from anime, realistic, cartoon, watercolor, and more
- **Low Latency**: Optimized for 20+ FPS real-time performance

## ✋ Gesture Guide

![Gesture Guide](images/Gesture_Guide.png)

| Gesture | Action | Description |
|---------|--------|-------------|
| ☝️ Index finger up | **Draw** | Move finger to draw strokes |
| ✌️ Index + Middle up | **Pause** | Stop drawing, move without marking |
| 👌 Thumb + Index pinch | **Submit** | Hold for 0.5s to generate AI image |
| ✊ Closed fist | **Clear** | Hold for 0.8s to clear canvas |
| 🤙 Pinky only | **Undo** | Undo last stroke |
| 🖐️ Three fingers | **Erase** | Erase strokes under finger |

## 🔄 Workflow

![Workflow](images/Workflow_image.png)

### Sketch to AI Transformation

| Input Sketch | AI Generated Output |
|:------------:|:-------------------:|
| ![Sketch](images/Sketch.png) | ![Generated](images/Sketch%20after%20Process.png) |

## 🛠️ Installation

### Prerequisites

- Python 3.9 or higher
- Webcam
- Windows/macOS/Linux

### Setup

1. **Clone the repository**
   ```bash
   cd c:\Users\Qazi\Downloads\Projects\Gesture+Gen
   ```

2. **Activate the virtual environment**
   ```powershell
   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up API key (required for AI generation)**
   ```powershell
   # Windows PowerShell
   $env:HF_TOKEN="your_huggingface_token"
   
   # Or permanently via System Environment Variables
   ```
   
   Get your free API token at [Hugging Face](https://huggingface.co/settings/tokens)
   
   **Recommended Models for Sketch-to-Image:**
   - `black-forest-labs/FLUX.1-schnell` - Fast, good quality (default)
   - `stabilityai/stable-diffusion-xl-base-1.0` - High quality
   - `tencent/HunyuanImage-3.0-Instruct` - Good for image editing

## 🚀 Running the Application

### Basic Usage

```bash
python main.py
```

### Command Line Options

```bash
python main.py --camera 0      # Use specific camera (default: 0)
python main.py --mock          # Use mock generator (no API needed)
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `1-8` | Select brush color |
| `+/-` | Increase/decrease brush size |
| `S` | Cycle through styles |
| `G` | Manually trigger generation |
| `U` | Undo last stroke |
| `R` | Redo |
| `C` | Clear canvas |
| `Space` | Save current sketch |
| `H` | Toggle generated image display |
| `Q` / `Esc` | Quit application |

## 📁 Project Structure

```
Gesture+Gen/
├── main.py                 # Application entry point
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── .env                   # Environment variables (secrets)
├── config.json            # Configuration settings
├── images/                # Documentation images
│   ├── Hero_Banner_image.png
│   ├── Gesture_Guide.png
│   ├── Workflow_image.png
│   ├── Sketch.png
│   └── Sketch after Process.png
├── models/                # Downloaded ML models
├── src/
│   ├── __init__.py        # Package initialization
│   ├── camera.py          # Webcam stream handler
│   ├── hand_tracking.py   # MediaPipe hand detection
│   ├── gesture_logic.py   # Gesture recognition
│   ├── canvas.py          # Virtual drawing board
│   ├── sketch_processor.py # Sketch preprocessing
│   ├── image_generator.py # AI generation backend
│   └── ui.py              # Main application UI
└── output/                # Saved sketches and images
```

## 🧠 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Gesture+Gen Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────┐   │
│  │  Webcam  │───▶│ Hand Tracker │───▶│ Gesture Detect  │   │
│  │  Camera  │    │  (MediaPipe) │    │  (Finger Logic) │   │
│  └──────────┘    └──────────────┘    └────────┬────────┘   │
│                                                │             │
│                                                ▼             │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────┐   │
│  │ Display  │◀───│    Canvas    │◀───│ Action Mapping  │   │
│  │  Output  │    │   (Drawing)  │    │ (Draw/Erase/etc)│   │
│  └──────────┘    └──────────────┘    └─────────────────┘   │
│       ▲                 │                                    │
│       │                 ▼                                    │
│       │         ┌──────────────┐    ┌─────────────────┐    │
│       └─────────│   Sketch     │───▶│  AI Generator   │    │
│                 │  Processor   │    │ (Stable Diffusion)│   │
│                 └──────────────┘    └─────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 🎨 Style Presets

| Style | Description |
|-------|-------------|
| `digital_art` | Clean digital illustration with vibrant colors |
| `anime` | Japanese animation style, cel-shaded |
| `realistic` | Photorealistic rendering |
| `cartoon` | Fun, colorful cartoon style |
| `sketch` | Pencil sketch, detailed line art |
| `watercolor` | Soft watercolor painting effect |
| `oil_painting` | Classical oil painting texture |

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_API_KEY` | Hugging Face API token | None (uses mock) |

### Camera Settings

Edit `src/camera.py` to adjust:
- Resolution (default: 1280x720)
- Frame rate (default: 30 FPS)
- Camera device index

### Gesture Sensitivity

Edit `src/gesture_logic.py` to adjust:
- `PINCH_THRESHOLD`: Sensitivity for pinch detection
- `STABILITY_TIME`: Time to hold gesture before activation
- `SUBMIT_HOLD_TIME`: Time to hold submit gesture

## 📊 Performance Tips

1. **Use good lighting** - Hand tracking works best with even lighting
2. **Keep hand in frame** - Ensure your hand is fully visible
3. **Clean background** - Avoid cluttered backgrounds
4. **Stable hand** - Hold gestures steady for best recognition
5. **GPU acceleration** - For local generation, use CUDA-enabled GPU

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- [MediaPipe](https://mediapipe.dev/) - Hand tracking
- [OpenCV](https://opencv.org/) - Computer vision
- [Hugging Face](https://huggingface.co/) - AI models and inference
- [Stable Diffusion](https://stability.ai/) - Image generation

---

Made with ❤️ by Gesture+Gen Team
