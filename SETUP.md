# 🚀 Violence Detection Setup Guide

## Quick Setup Steps

### 1️⃣ Install Dependencies

```bash
cd violence_detection_test
pip install -r requirements.txt
```

### 2️⃣ Configure API Key

Edit the `.env` file (already created for you):

```bash
nano .env
```

**Replace** `your_api_key_here` with your actual Gemini API key:

```
GEMINI_API_KEY=AIzaSyDHFaoEGCWjX9MrYkzi_4vFplUpBhZNFCs
```

### 3️⃣ Add Test Video

Place your video file in the `videos/` folder:

```bash
cp /path/to/your/Fighting027_x264.mp4 videos/
```

Then update the video path in `.env`:

```
VIDEO_PATH=videos/Fighting027_x264.mp4
```

### 4️⃣ Run the System

```bash
./run.sh
```

Or manually:

```bash
python3 main.py
```

---

## 📝 What to Expect

### System Startup

```
🚨 REAL-TIME VIOLENCE DETECTION SYSTEM
════════════════════════════════════════════════════════════
⚙️  CONFIGURATION
════════════════════════════════════════════════════════════
Device: cuda
Edge Model: vikhyatk/moondream2
Cloud Model: gemini-flash-lite-latest
Video: Fighting027_x264.mp4
Motion Threshold: 2000
Max Investigation Rounds: 10
```

### During Operation

```
🎥 [CCTV] Monitoring Feed: Fighting027_x264.mp4
   ... [CCTV] 00:00:01 ...
   ... [CCTV] 00:00:02 ...

📹 [MOTION DETECTED] at 00:01:15

🔍 [INVESTIGATION] Started at 00:01:15
📸 [EDGE @00:01:15] Analyzing initial frame...
   Response: Two people in aggressive stance, physical contact
☁️  [CLOUD] Analyzing initial report...
   Status: INVESTIGATE | Confidence: 65%
   Reason: Suspicious activity, need more details
```

### Final Verdict

```
🚨🚨🚨 [VERDICT] VIOLENCE DETECTED! 🚨🚨🚨
   Confidence: 87%
   Reason: Confirmed physical violence with aggressive actions
   Rounds of investigation: 3
```

---

## ⚠️ Troubleshooting

### "API key not set"

- Make sure you edited `.env` and added your real API key
- No quotes needed around the API key

### "Video file not found"

- Check the path in `.env` matches your video location
- Use relative path: `videos/your_video.mp4`

### First run is slow

- AI models download on first run (~4GB)
- This only happens once, they're cached after

---

## 📚 Full Documentation

See `README.md` for complete documentation including:

- Architecture details
- Configuration options
- Advanced troubleshooting
- Project structure

---

✅ **Ready to test!** Follow the 4 steps above and you're good to go! 🚀
