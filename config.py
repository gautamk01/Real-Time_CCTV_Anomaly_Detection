"""Configuration Management for Violence Detection System."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration manager using environment variables."""
    
    # Project paths
    BASE_DIR = Path(__file__).parent
    VIDEOS_DIR = BASE_DIR / "videos"
    
    # API Configuration
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

    # Model Configuration
    EDGE_MODEL_ID = "vikhyatk/moondream2"
    CLOUD_MODEL_ID = "llama-3.3-70b-versatile"
    
    # Device Configuration
    DEVICE = os.getenv("DEVICE", "auto")  # "cuda", "cpu", or "auto"
    
    # Video Configuration
    VIDEO_PATH = os.getenv("VIDEO_PATH", str(VIDEOS_DIR / "test_video.mp4"))
    
    # Detection Parameters
    MOTION_THRESHOLD = int(os.getenv("MOTION_THRESHOLD", "2000"))
    MAX_INVESTIGATION_ROUNDS = int(os.getenv("MAX_INVESTIGATION_ROUNDS", "3"))
    
    # Alert Saving
    SAVE_ALERTS = os.getenv("SAVE_ALERTS", "true").lower() == "true"
    ALERTS_DIR = BASE_DIR / os.getenv("ALERTS_DIR", "alerts")
    BUFFER_DURATION_SECONDS = int(os.getenv("BUFFER_DURATION_SECONDS", "10"))
    CLEANUP_OLD_ALERTS_DAYS = int(os.getenv("CLEANUP_OLD_ALERTS_DAYS", "7"))
    
    # Firebase Cloud Messaging
    ENABLE_FCM_NOTIFICATIONS = os.getenv("ENABLE_FCM_NOTIFICATIONS", "false").lower() == "true"
    FIREBASE_CREDENTIALS_PATH = BASE_DIR / os.getenv("FIREBASE_CREDENTIALS_PATH", "firebase_key.json")
    FCM_TOPIC = os.getenv("FCM_TOPIC", "violence_alerts")

    
    # A2A (Agent-to-Agent) Communication
    ENABLE_A2A = os.getenv("ENABLE_A2A", "false").lower() == "true"
    EDGE_AGENT_URL = os.getenv("EDGE_AGENT_URL", "http://localhost:8001")
    CLOUD_AGENT_URL = os.getenv("CLOUD_AGENT_URL", "http://localhost:8002")
    RAG_AGENT_URL = os.getenv("RAG_AGENT_URL", "http://localhost:8003")

    # RAG (Retrieval-Augmented Generation)
    ENABLE_RAG = os.getenv("ENABLE_RAG", "true").lower() == "true"
    RAG_DB_DIR = BASE_DIR / os.getenv("RAG_DB_DIR", "rag_db")
    RAG_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))

    # Frame Processing
    MOTION_BLUR_KERNEL = (5, 5)
    MOTION_BINARY_THRESHOLD = 20
    MOTION_DILATE_ITERATIONS = 3
    
    @classmethod
    def validate(cls):
        """Validate critical configuration values."""
        errors = []
        
        if not cls.GROQ_API_KEY:
            errors.append("❌ GROQ_API_KEY not set in .env file")
        
        if not Path(cls.VIDEO_PATH).exists():
            errors.append(f"❌ Video file not found: {cls.VIDEO_PATH}")
        
        if errors:
            print("\n" + "="*60)
            print("🔴 CONFIGURATION ERRORS")
            print("="*60)
            for error in errors:
                print(error)
            print("\nPlease check your .env file and video path.")
            print("="*60 + "\n")
            return False
        
        return True
    
    @classmethod
    def print_info(cls):
        """Print configuration information."""
        import torch
        
        device = cls.DEVICE
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print("\n" + "="*60)
        print("⚙️  CONFIGURATION")
        print("="*60)
        print(f"Device: {device}")
        print(f"Edge Model: {cls.EDGE_MODEL_ID}")
        print(f"Cloud Model: {cls.CLOUD_MODEL_ID}")
        print(f"Video: {Path(cls.VIDEO_PATH).name}")
        print(f"Motion Threshold: {cls.MOTION_THRESHOLD}")
        print(f"Max Investigation Rounds: {cls.MAX_INVESTIGATION_ROUNDS}")
        print(f"Save Alerts: {'Yes' if cls.SAVE_ALERTS else 'No'}")
        if cls.SAVE_ALERTS:
            print(f"Alerts Directory: {cls.ALERTS_DIR}")
            print(f"Buffer Duration: {cls.BUFFER_DURATION_SECONDS}s")
        print(f"A2A Mode: {'Networked' if cls.ENABLE_A2A else 'Direct (local)'}")
        if cls.ENABLE_A2A:
            print(f"Edge Agent: {cls.EDGE_AGENT_URL}")
            print(f"Cloud Agent: {cls.CLOUD_AGENT_URL}")
            print(f"RAG Agent: {cls.RAG_AGENT_URL}")
        print(f"RAG: {'Enabled' if cls.ENABLE_RAG else 'Disabled'}")
        print("="*60 + "\n")
