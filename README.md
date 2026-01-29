# 🚨 Real-Time Violence Detection System

AI-powered violence detection using hybrid edge-cloud architecture with motion-triggered investigation.

## 🎯 Features

- **Motion-Triggered Analysis**: Only investigates when motion is detected
- **Edge AI**: Moondream2 local vision model for frame analysis
- **Cloud AI**: Gemini for high-level threat assessment
- **Real-Time Processing**: Multi-threaded architecture with live frame access
- **Conversation Loop**: Iterative investigation with dynamic questioning

## 🏗️ Architecture

```
Video Feed → Motion Detection → Edge Vision Analysis
                                        ↓
                                  Cloud Assessment
                                        ↓
                    ┌─────────────────┴──────────────┐
                    ↓                                ↓
                 CLEAR                          INVESTIGATE
            (Back to monitoring)           (Analyze current frame)
                                                    ↓
                                              Cloud decides
                                                    ↓
                                    ┌──────────────┴──────────┐
                                    ↓                         ↓
                                 CLEAR                     ALERT
                                                    (Violence detected!)
```

## 📋 Prerequisites

- Python 3.8+
- CUDA-enabled GPU (optional, but recommended)
- Google Gemini API key
- Test video file (MP4/AVI)

## 🚀 Installation

### 1. Install Dependencies

```bash
cd violence_detection_test
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy template
cp .env.example .env

# Edit .env and set:
# - GEMINI_API_KEY: Your Google Gemini API key
# - VIDEO_PATH: Path to your test video
```

### 3. Add Test Video

Place your test video in the `videos/` directory:

```bash
cp /path/to/your/video.mp4 videos/test_video.mp4
```

## 🎬 Usage

### Quick Start

```bash
./run.sh
```

### Manual Start

```bash
python3 main.py
```

### Example .env Configuration

```env
GEMINI_API_KEY=AIzaSyDHFaoEGCWjX9MrYkzi_4vFplUpBhZNFCs
VIDEO_PATH=videos/fighting_scene.mp4
DEVICE=auto
MOTION_THRESHOLD=2000
MAX_INVESTIGATION_ROUNDS=10
```

## 📊 Understanding Output

### Motion Detection

```
📹 [MOTION DETECTED] at 00:01:23
```

### Investigation Start

```
🔍 [INVESTIGATION] Started at 00:01:23
📸 [EDGE @00:01:23] Analyzing initial frame...
   Response: Two people in close proximity, physical contact visible
☁️  [CLOUD] Analyzing initial report...
   Status: INVESTIGATE | Confidence: 60%
   Reason: Physical contact detected, need more context
```

### Investigation Loop

```
☁️  [CLOUD] Asking: "Are the people showing aggressive body language?"
📸 [EDGE @00:01:24] Analyzing CURRENT frame...
   Response: Yes, aggressive postures with raised arms
☁️  [CLOUD] Analyzing update (Round 2)...
   Status: ALERT | Confidence: 85%
```

### Final Verdict

```
🚨🚨🚨 [VERDICT] VIOLENCE DETECTED! 🚨🚨🚨
   Confidence: 85%
   Reason: Aggressive physical interaction confirmed
   Rounds of investigation: 3
```

## ⚙️ Configuration Options

| Variable                   | Default | Description                                |
| -------------------------- | ------- | ------------------------------------------ |
| `DEVICE`                   | `auto`  | Hardware to use (`cuda`, `cpu`, or `auto`) |
| `MOTION_THRESHOLD`         | `2000`  | Minimum contour area to trigger motion     |
| `MAX_INVESTIGATION_ROUNDS` | `10`    | Max iterations before investigation ends   |

## 🧪 Testing

### 1. Verify Installation

```bash
python3 -c "import torch; import transformers; import google.generativeai; print('✅ All modules loaded')"
```

### 2. Check GPU Availability

```bash
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 3. Test Configuration

```bash
python3 -c "from config import Config; Config.validate()"
```

## 🐛 Troubleshooting

### API Key Error

```
❌ GEMINI_API_KEY not set in .env file
```

**Solution**: Create `.env` file and add your API key

### Video Not Found

```
❌ Video file not found: videos/test_video.mp4
```

**Solution**: Ensure video file exists at specified path

### CUDA Out of Memory

```
CUDA out of memory
```

**Solution**: Set `DEVICE=cpu` in `.env` file

### Model Download Slow

Models will download on first run. This can take several minutes:

- Moondream2: ~4GB
- Be patient, downloaded once and cached locally

## 📁 Project Structure

```
violence_detection_test/
├── main.py                 # Entry point
├── config.py              # Configuration manager
├── models/
│   ├── edge_vision.py    # Moondream2 wrapper
│   └── cloud_ai.py       # Gemini wrapper
├── core/
│   ├── frame_buffer.py   # Thread-safe frame storage
│   ├── motion_detector.py # Motion detection
│   └── investigator.py   # AI investigation logic
├── videos/               # Test videos
├── requirements.txt      # Dependencies
├── .env.example         # Configuration template
└── run.sh               # Quick start script
```

## 🔐 Security Notes

- ⚠️ Never commit `.env` file (contains API key)
- `.gitignore` protects sensitive files automatically
- Keep API key secure and rotate if exposed

## 📝 License

This is a testing application for research purposes.

## 🤝 Support

For issues or questions:

1. Check troubleshooting section
2. Verify all dependencies installed
3. Ensure API key is valid
4. Check video file format (MP4/AVI recommended)
