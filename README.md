# 🚨 Real-Time Violence Detection System

**Distributed GPU Inference Architecture with Temporal Escalation**

An AI-powered real-time surveillance system that detects violent behavior in CCTV footage using a hybrid edge-cloud architecture. The system employs a two-agent investigation pipeline (local vision + cloud reasoning) with temporal escalation — designed to **eliminate manual review** and minimize false positives.

---

## 🎯 Key Features

### Core Detection Pipeline
- **Motion-Gated Analysis** — Only activates AI when persistent motion is detected (consecutive frame gating), saving GPU cycles on static scenes
- **Edge AI (Moondream2)** — Local vision model runs on GPU for sub-2s frame analysis with 8-bit quantization
- **Cloud AI (Groq LPU)** — Large language model performs high-level threat reasoning with sub-second inference
- **Multi-Round Investigation** — Iterative edge↔cloud conversation with open-ended questioning to avoid confirmation bias from yes/no prompts

### Temporal Escalation Engine
- **Cross-Investigation Memory** — Rolling verdict tracker per camera detects persistent incidents that no single investigation can catch
- **Automatic Escalation** — 2+ consecutive suspicious verdicts within 120s → **ALERT** (no manual review needed)
- **Adaptive Cooldown** — CLEAR scenes get long cooldowns (10s), suspicious scenes get short cooldowns (1-3s), ALERTs get immediate re-monitoring (1s)
- **High-Confidence CLEAR Reset** — Only strong CLEARs (≥80% confidence) reset the escalation counter; timeout fallbacks don't

### False Positive Reduction
- **Surveillance Context Rules** — Cloud AI understands it's monitoring public spaces: "kicking" = violence, not sparring practice
- **Non-Real Imagery Filter** — Automatically dismisses cartoons, posters, artwork, TV screens, and decorations as CLEAR
- **Anti-Repetition Logic** — Tracks questions already asked per investigation; forces the cloud to vary its questioning angle each round
- **Cascading Confidence Scoring** — Edge-only fallback for cloud timeouts uses weighted keyword scoring with timeout penalty (capped at REVIEW, never auto-ALERTs)

### Multi-Camera Support
- **Independent Per-Camera Pipelines** — Each camera has its own investigation thread, escalation state, and cooldown timer
- **Shared GPU Queue** — Single inference server handles all cameras via thread-safe priority queue (O(1) slot pattern)
- **Per-Camera Overrides** — Individual motion thresholds, cooldowns, and motion gating per camera
- **Throughput Mode** — Auto-caps investigation rounds when multiple cameras are active

### Distributed Architecture
- **FPC-ARC (Freshness-Priority Coalescing with Adaptive Rate Control)** — Atomic slot pattern ensures the system always processes the freshest frame, not stale queued events
- **GPU Warm-Up** — Mandatory dummy forward pass at startup eliminates first-inference latency and cold-start race conditions
- **Pipeline Parallelism** — While the cloud processes round N, the GPU pre-encodes the next frame (overlaps GPU compute with network I/O)
- **Slot Overwrite Pattern** — New motion events atomically replace old pending events, bounding memory to O(1) per camera

### Alert & Notification System
- **Firebase Cloud Messaging (FCM)** — Push notifications to mobile devices with alert details
- **Alert Evidence Packages** — Saves buffered frames, investigation history, and verdict metadata for each incident
- **Escalation Alerts** — Temporal escalation triggers are saved separately with full verdict chain context
- **Benchmark Metrics** — JSON/CSV export of latency, confidence, token usage, memory stats per run

### RAG (Retrieval-Augmented Generation)
- **Knowledge Base** — Past investigation verdicts are ingested into a vector database (ChromaDB + sentence-transformers)
- **Context Injection** — Cloud AI receives relevant past incidents as context when assessing new scenes
- **Auto-Ingest** — Every investigation result is automatically added to the knowledge base

### Agent-to-Agent (A2A) Communication
- **Direct Mode** — Edge and cloud models communicate directly within the same process (zero-overhead IPC)
- **Server Mode** — Optional HTTP-based agent servers for distributed deployment across machines
- **Protocol Layer** — Standardized message format for inter-agent communication

---

## 🏗️ Architecture

```
Camera Feeds (N cameras)
    │
    ▼
┌─────────────────────────────────────────────────┐
│              CameraManager                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │ cam1     │  │ cam2     │  │ cam3     │     │
│  │ capture  │  │ capture  │  │ capture  │     │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘     │
│       │              │              │           │
│       ▼              ▼              ▼           │
│  Motion Gate    Motion Gate    Motion Gate      │
│       │              │              │           │
│       ▼              ▼              ▼           │
│  ┌────────────────────────────────────────┐     │
│  │     Slot Pattern (1 pending per cam)   │     │
│  └────────────────┬───────────────────────┘     │
└───────────────────┼─────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│           CameraPipeline (per camera)           │
│                                                 │
│  ┌──────────────────────────────────┐          │
│  │     AIInvestigator               │          │
│  │  ┌─────────┐    ┌───────────┐   │          │
│  │  │ Edge AI │◄──►│ Cloud AI  │   │          │
│  │  │(GPU,local)   │(Groq,cloud)   │          │
│  │  └────┬────┘    └─────┬─────┘   │          │
│  │       │  Multi-Round  │         │          │
│  │       │  Investigation│         │          │
│  │       └───────┬───────┘         │          │
│  └───────────────┼─────────────────┘          │
│                  ▼                             │
│  ┌──────────────────────────────────┐          │
│  │     EscalationTracker            │          │
│  │  verdict → counter → ESCALATE?   │          │
│  └──────────────┬───────────────────┘          │
│                 │                               │
│         ┌───────┴───────┐                      │
│         ▼               ▼                      │
│      CLEAR          ALERT                      │
│  (keep monitoring)  (save + notify)            │
└─────────────────────────────────────────────────┘
```

### Verdict Flow

| Verdict | Meaning | User-Facing? | Action |
|---------|---------|:------------:|--------|
| **CLEAR** | No threat | ✅ | Reset escalation, long cooldown |
| **INVESTIGATE** | Need more info | ❌ (internal) | Ask edge another question |
| **REVIEW** | Ambiguous scene | ❌ (internal) | Feed escalation tracker silently |
| **ALERT** | Violence confirmed | ✅ | Save evidence, FCM push, 1s cooldown |
| **ESCALATION** | Persistent incident | ✅ (as ALERT) | Same as ALERT — triggered by temporal pattern |

---

## 📋 Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (GTX 1660 Ti or better recommended)
- Groq API key (for cloud inference) or Ollama (local alternative)
- Test video files (MP4/AVI)

---

## 🚀 Installation

### 1. Clone & Setup Virtual Environment

```bash
cd surveillance_system
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Cloud AI Backend: "groq" or "ollama"
CLOUD_BACKEND=groq
GROQ_API_KEY=your_groq_api_key_here
CLOUD_MODEL_ID=llama-3.3-70b-versatile

# Device (cuda, cpu, or auto)
DEVICE=auto

# Single camera mode
VIDEO_PATH=test_video.mp4

# Multi-camera mode (overrides VIDEO_PATH)
CAMERA_SOURCES=cam1:test_video_2.mp4,cam2:test_video_3.mp4

# Detection tuning
MOTION_THRESHOLD=2000
MIN_CONSECUTIVE_MOTION_FRAMES=2
MAX_INVESTIGATION_ROUNDS=3
CAMERA_COOLDOWN=2.0

# Alert saving
SAVE_ALERTS=true
ALERTS_DIR=alerts
BUFFER_DURATION_SECONDS=10

# Firebase push notifications (optional)
ENABLE_FCM_NOTIFICATIONS=false
FIREBASE_CREDENTIALS_PATH=firebase_key.json
FCM_TOPIC=violence_alerts

# RAG knowledge base
ENABLE_RAG=true
```

---

## 🎬 Usage

### Quick Start

```bash
python main.py
```

### Multi-Camera Mode

Set `CAMERA_SOURCES` in `.env`:
```env
CAMERA_SOURCES=cam1:test_video_2.mp4,cam2:test_video_3.mp4
```

Supports: video files, device IDs (`/dev/video0`), RTSP streams (`rtsp://host:8554/stream`)

### Per-Camera Overrides

```env
CAMERA_OVERRIDES=cam1:threshold=2500,cooldown=2.5;cam2:threshold=1800,min_motion_frames=3
```

### Switching Cloud Backend

**Groq (cloud — recommended):**
```env
CLOUD_BACKEND=groq
GROQ_API_KEY=your_key
CLOUD_MODEL_ID=llama-3.3-70b-versatile
```

**Ollama (local):**
```env
CLOUD_BACKEND=ollama
CLOUD_MODEL_ID=qwen3:8b
```

---

## 📊 Understanding Output

### Normal Flow — No Threat
```
📸 [cam1] [EDGE @00:00:05] Analyzing initial frame...
   Response: Two people talking at a table.
☁️  [cam1] [CLOUD] Analyzing initial report...
   Status: CLEAR | Confidence: 90%

✅ [cam1] [VERDICT] ALL CLEAR - No threat detected
```

### Investigation Flow — Ambiguous Scene
```
📸 [cam1] [EDGE @00:00:14] Analyzing initial frame...
   Response: A person appears to be falling near the counter.
☁️  [cam1] [CLOUD] Analyzing initial report...
   Status: INVESTIGATE | Confidence: 65%

☁️  [cam1] [CLOUD] Asking: "Describe the body position of every person within 2 meters..."
📸 [cam1] [EDGE @00:00:16] Answering with pre-encoded frame...
   Status: REVIEW | Confidence: 65%

🔍 [cam1] [MONITORING] Scene unclear — tracking for escalation
   [ESCALATION cam1] 1 consecutive suspicious (REVIEW) (need 2 to escalate)
```

### Escalation — Persistent Threat Detected
```
############################################################
[ESCALATION cam1] 🚨 TEMPORAL ESCALATION TRIGGERED
   Suspicious verdicts: 3 in 18s
   Max confidence seen: 75%
############################################################

🚨🚨🚨 [cam1] [VERDICT] ESCALATION ALERT — PERSISTENT THREAT DETECTED 🚨🚨🚨
   Confidence: 85%
   Reason: Persistent incident: Camera cam1 flagged 3 suspicious verdicts in 18s
```

### Direct Alert — Violence Caught on Frame
```
📸 [cam1] [EDGE @00:00:42] Analyzing initial frame...
   Response: One person appears to be kicking the other person.
☁️  [cam1] [CLOUD] Analyzing initial report...
   Status: ALERT | Confidence: 100%

🚨🚨🚨 [cam1] [VERDICT] VIOLENCE DETECTED! 🚨🚨🚨
   Confidence: 100%
   Reason: Active violence confirmed: person kicking another person
```

---

## ⚙️ Configuration Reference

### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `auto` | `cuda`, `cpu`, or `auto` |
| `CLOUD_BACKEND` | `groq` | `groq` (cloud) or `ollama` (local) |
| `CLOUD_MODEL_ID` | `llama-3.3-70b-versatile` | Cloud LLM model ID |
| `EDGE_QUANT_MODE` | `auto` | Edge model quantization: `auto`, `4bit`, `8bit` |

### Detection Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `MOTION_THRESHOLD` | `2000` | Min contour area to trigger motion |
| `MIN_CONSECUTIVE_MOTION_FRAMES` | `2` | Consecutive frames needed before investigation |
| `MAX_INVESTIGATION_ROUNDS` | `3` | Max edge↔cloud conversation rounds |
| `CAMERA_COOLDOWN` | `2.0` | Base cooldown between investigations (seconds) |

### Multi-Camera

| Variable | Default | Description |
|----------|---------|-------------|
| `CAMERA_SOURCES` | *(empty)* | Comma-separated `id:path` pairs |
| `CAMERA_OVERRIDES` | *(empty)* | Per-camera threshold/cooldown overrides |
| `MULTI_CAMERA_MAX_INVESTIGATION_ROUNDS` | `2` | Round cap when multiple cameras active |
| `GPU_QUEUE_SIZE` | `32` | Shared GPU inference queue capacity |

### Alerts & Notifications

| Variable | Default | Description |
|----------|---------|-------------|
| `SAVE_ALERTS` | `true` | Save evidence frames and metadata |
| `ALERTS_DIR` | `alerts` | Directory for saved alert packages |
| `BUFFER_DURATION_SECONDS` | `10` | Frame buffer duration for evidence capture |
| `ENABLE_FCM_NOTIFICATIONS` | `false` | Firebase push notifications |
| `FCM_TOPIC` | `violence_alerts` | FCM topic for alert broadcasts |

### RAG Knowledge Base

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_RAG` | `true` | Enable past-investigation context retrieval |
| `RAG_DB_DIR` | `rag_db` | ChromaDB storage directory |
| `RAG_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model for embeddings |
| `RAG_TOP_K` | `3` | Number of past incidents to retrieve as context |

### Edge Model Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `EDGE_INITIAL_MAX_TOKENS` | `24` | Max tokens for initial frame description |
| `EDGE_FOLLOWUP_MAX_TOKENS` | `16` | Max tokens for follow-up answers |

---

## 📁 Project Structure

```
violence_detection_system/
├── main.py                      # Entry point — startup, display, shutdown
├── config.py                    # Configuration manager (env vars)
├── requirements.txt             # Python dependencies
├── .env.example                 # Configuration template
├── run.sh                       # Quick start script
│
├── core/                        # Core detection pipeline
│   ├── camera_manager.py        # Multi-camera source manager
│   ├── camera_pipeline.py       # Per-camera FPC-ARC pipeline
│   ├── escalation_tracker.py    # Temporal escalation engine
│   ├── investigator.py          # Edge↔cloud investigation loop
│   ├── inference_server.py      # Shared GPU inference queue
│   ├── motion_detector.py       # Motion detection with gating
│   ├── frame_buffer.py          # Thread-safe circular frame buffer
│   ├── alert_saver.py           # Evidence package saver
│   ├── fcm_notifier.py          # Firebase push notifications
│   ├── metrics_logger.py        # Benchmark metrics collection
│   └── evaluation.py            # Detection evaluation tools
│
├── models/                      # AI model wrappers
│   ├── edge_vision.py           # Moondream2 wrapper (GPU inference)
│   ├── cloud_ai.py              # Cloud LLM wrapper (Groq/Ollama)
│   └── examples_database.py     # Few-shot examples for cloud prompt
│
├── a2a/                         # Agent-to-Agent communication
│   ├── client.py                # A2A client (direct/server mode)
│   ├── edge_agent.py            # Edge agent server
│   ├── cloud_agent.py           # Cloud agent server
│   ├── launcher.py              # Agent server launcher
│   └── protocol.py              # Inter-agent message protocol
│
├── rag/                         # Retrieval-Augmented Generation
│   ├── knowledge_base.py        # ChromaDB vector store
│   ├── rag_agent.py             # RAG query agent
│   └── bootstrap.py             # Initial knowledge base seeding
│
├── alerts/                      # Saved alert evidence packages
├── rag_db/                      # Persistent vector database
├── docs/                        # Documentation & architecture plans
├── flutter_app/                 # Mobile companion app
└── videos/                      # Test video files
```

---

## 🧪 Testing

### Verify Installation

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

### Run with Test Video

```bash
# Single camera
VIDEO_PATH=test_video_2.mp4 python main.py

# Multi-camera
CAMERA_SOURCES="cam1:test_video_2.mp4,cam2:test_video_3.mp4" python main.py
```

### Check Benchmark Results

```bash
cat alerts/benchmark_metrics.json | python -m json.tool
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| `CUDA out of memory` | Set `DEVICE=cpu` or reduce `GPU_QUEUE_SIZE` |
| Cloud timeouts (10s) | Switch to `llama-3.3-70b-versatile` (most reliable on Groq) |
| Model download slow | First run downloads ~4GB. Cached after that |
| Groq rate limit | Switch to Ollama: `CLOUD_BACKEND=ollama` |
| False positives on posters | The system auto-filters cartoons/artwork — check if edge model mentions them |
| Too many investigations | Increase `MOTION_THRESHOLD` or `MIN_CONSECUTIVE_MOTION_FRAMES` |

---

## 🔐 Security Notes

- ⚠️ Never commit `.env` — contains API keys
- `.gitignore` protects sensitive files automatically
- Firebase credentials (`.json`) should be kept secure
- Alert evidence packages may contain sensitive footage

---

## 📝 Algorithms

### FPC-ARC (Freshness-Priority Coalescing with Adaptive Rate Control)
Each camera uses an atomic single-slot pattern instead of a FIFO queue. New motion events overwrite stale ones, guaranteeing the investigation always analyzes the freshest frame. Adaptive cooldowns (AIMD-inspired) modulate investigation frequency based on verdict severity.

### Temporal Escalation Engine
Per-camera rolling verdict tracker. If a camera produces ≥2 consecutive suspicious (non-high-confidence-CLEAR) verdicts within a 120-second window, the system escalates to ALERT — even if no single investigation could confirm violence individually. This eliminates the "perpetual REVIEW" problem for persistent incidents.

### Cascading Confidence Scoring (Edge Fallback)
When the cloud times out, the system falls back to weighted keyword scoring of edge descriptions:
1. Score each observation using alert/review/clear keyword categories
2. Combine scores using probability union
3. Apply timeout uncertainty penalty (0.7×)
4. Cap at REVIEW — never auto-ALERTs without cloud confirmation

---

## 📄 License

Research prototype — Patent pending.

---

## 🤝 Contributing

For issues or questions:
1. Check the troubleshooting table
2. Review `alerts/benchmark_metrics.json` for performance data
3. Ensure your GPU has ≥4GB VRAM for Moondream2 (8-bit)
