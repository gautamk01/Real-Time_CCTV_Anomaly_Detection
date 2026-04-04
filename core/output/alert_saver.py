"""Incident frame saving - thread-based I/O for zero latency."""

import json
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from PIL import Image


def save_alert_sync(frames_data: List[Dict], incident_dir: Path,
                    verdict: Dict, history: List[str]):
    """
    Save alert incident to disk (runs in background thread).
    
    Args:
        frames_data: List of dicts with 'frame' (PIL Image), 'timestamp', 'motion'
        incident_dir: Directory to save this incident
        verdict: Investigation verdict dict
        history: Investigation history log
    """
    try:
        # Create incident directory
        incident_dir.mkdir(parents=True, exist_ok=True)
        
        # Save frames
        for idx, frame_data in enumerate(frames_data):
            frame_path = incident_dir / f"frame_{idx:04d}_{frame_data['timestamp']}.jpg"
            frame_data['frame'].save(frame_path, 'JPEG', quality=90)
        
        # Save metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'verdict': verdict,
            'history': history,
            'frame_count': len(frames_data),
            'timestamps': [f['timestamp'] for f in frames_data]
        }
        
        metadata_path = incident_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 [SAVED] Alert incident saved to {incident_dir.name}")
        
    except Exception as e:
        print(f"❌ [SAVE ERROR] Failed to save alert: {e}")


def save_alert_incident(frames_data: List[Dict], alerts_dir: Path,
                        verdict: Dict, history: List[str],
                        prefix: str = "alert") -> str:
    """
    Trigger background save without blocking (fire-and-forget).
    
    Args:
        frames_data: Frames to save
        alerts_dir: Base alerts directory
        verdict: Investigation verdict  
        history: Investigation history
        prefix: Directory prefix, e.g. "alert" or "review"

    Returns:
        The created incident directory name
    """
    # Create unique incident directory name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    confidence = verdict.get('confidence', 0)
    incident_name = f"{prefix}_{timestamp}_conf{confidence}"
    incident_dir = alerts_dir / incident_name
    
    # Run save in background thread (non-blocking, no event loop needed)
    save_thread = threading.Thread(
        target=save_alert_sync,
        args=(frames_data, incident_dir, verdict, history),
        daemon=True  # Thread will auto-terminate when main program exits
    )
    save_thread.start()
    return incident_name
