"""Real-Time CCTV Violence Detection System."""

import os
import sys
import cv2
import time
import threading
from pathlib import Path
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from models import EdgeVision, CloudAI
from core import FrameBuffer, MotionDetector, AIInvestigator


# Global shared resources
frame_buffer = FrameBuffer()
stop_event = threading.Event()
models_ready = threading.Event()  # Signal when AI models are loaded


def ai_worker_loop(config: Config):
    """
    Background thread that monitors for motion and triggers investigation.
    Uses REAL-TIME frame access for each Cloud question.
    """
    print("\n🔧 [AI THREAD] Initializing models...")
    print("⏳ [AI THREAD] Please wait, this may take a few minutes on first run...")
    
    # Initialize AI models
    edge_vision = EdgeVision(
        model_id=config.EDGE_MODEL_ID,
        device=config.DEVICE
    )
    
    cloud_ai = CloudAI(
        api_key=config.GEMINI_API_KEY,
        model_id=config.CLOUD_MODEL_ID
    )
    
    # Get video FPS for buffer sizing
    cap_temp = cv2.VideoCapture(config.VIDEO_PATH)
    fps = cap_temp.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
    cap_temp.release()
    
    # Initialize FCM notifier if enabled
    fcm_notifier = None
    if config.ENABLE_FCM_NOTIFICATIONS:
        try:
            from core.fcm_notifier import FCMNotifier
            fcm_notifier = FCMNotifier(
                credentials_path=config.FIREBASE_CREDENTIALS_PATH,
                topic=config.FCM_TOPIC
            )
        except Exception as e:
            print(f"⚠️  [FCM] Could not initialize: {e}")
            print(f"   Continuing without mobile notifications")
    
    # Create investigator
    investigator = AIInvestigator(
        edge_vision=edge_vision,
        cloud_ai=cloud_ai,
        frame_provider=lambda: frame_buffer.get_current_frame(),
        max_rounds=config.MAX_INVESTIGATION_ROUNDS,
        fps=fps,
        save_alerts=config.SAVE_ALERTS,
        alerts_dir=config.ALERTS_DIR,
        buffer_duration=config.BUFFER_DURATION_SECONDS,
        fcm_notifier=fcm_notifier
    )
    
    print("\n✅ [AI THREAD] Ready to monitor for threats")
    
    # Signal that models are ready
    models_ready.set()
    
    while not stop_event.is_set():
        # Wait for motion detection signal
        if frame_buffer.wait_for_motion(timeout=0.1):
            # Get the frame that triggered motion
            initial_frame, initial_ts = frame_buffer.get_current_frame()
            
            if initial_frame is not None:
                print(f"\n{'#'*60}")
                print(f"📹 [MOTION DETECTED] at {initial_ts}")
                print(f"{'#'*60}")
                
                # Run real-time investigation
                result = investigator.investigate_realtime(
                    initial_frame,
                    initial_ts
                )
                
                # Clear motion flag so we can detect new events
                frame_buffer.clear_motion_flag()
                
                # Small cooldown to avoid re-triggering immediately
                time.sleep(1.0)


def video_capture_loop(config: Config):
    """
    Main thread: Continuously capture frames and update shared buffer.
    Detect motion and signal AI thread.
    """
    # Wait for AI models to be ready before starting video
    print("\n⏳ [CCTV] Waiting for AI models to load...")
    models_ready.wait()  # Block until models are ready
    print("✅ [CCTV] AI models ready, starting video playback\n")
    
    cap = cv2.VideoCapture(config.VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"❌ Failed to open video: {config.VIDEO_PATH}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
    frame_delay = 1.0 / fps
    
    print(f"\n🎥 [CCTV] Monitoring Feed: {Path(config.VIDEO_PATH).name}")
    print(f"⏱️  Sync Speed: {fps:.2f} FPS")
    print("="*60)
    
    # Initialize motion detector
    motion_detector = MotionDetector(
        threshold=config.MOTION_THRESHOLD,
        blur_kernel=config.MOTION_BLUR_KERNEL,
        binary_threshold=config.MOTION_BINARY_THRESHOLD,
        dilate_iterations=config.MOTION_DILATE_ITERATIONS
    )
    
    frame_count = 0
    
    while cap.isOpened() and not stop_event.is_set():
        loop_start = time.time()
        
        ret, frame_bgr = cap.read()
        if not ret:
            break
        
        # Calculate timestamp
        secs = frame_count / fps
        timestamp = time.strftime("%H:%M:%S", time.gmtime(secs))
        
        # Detect motion
        motion = motion_detector.detect(frame_bgr)
        
        # Convert to RGB PIL Image
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Update shared frame buffer
        frame_buffer.update_frame(pil_image, timestamp, motion=motion)
        
        # Display video window in real-time
        display_frame = frame_bgr.copy()
        
        # Add timestamp overlay
        cv2.putText(display_frame, timestamp, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add motion indicator
        if motion:
            cv2.putText(display_frame, "MOTION DETECTED", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Show video window
        cv2.imshow('Violence Detection - Live Feed', display_frame)
        
        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n⚠️  [USER] Quit requested")
            stop_event.set()
            break
        
        frame_count += 1
        
        # Real-time sync
        processing_time = time.time() - loop_start
        sleep_needed = frame_delay - processing_time
        if sleep_needed > 0:
            time.sleep(sleep_needed)
        
        # Progress indicator every second
        if frame_count % int(fps) == 0:
            print(f"   ... [CCTV] {timestamp} ...")
    
    cap.release()
    cv2.destroyAllWindows()
    stop_event.set()
    print("\n🎥 [CCTV] Video ended.")


def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("🚨 REAL-TIME VIOLENCE DETECTION SYSTEM")
    print("="*60)
    
    # Validate configuration
    if not Config.validate():
        sys.exit(1)
    
    # Print configuration
    Config.print_info()
    
    # Start AI worker thread
    ai_thread = threading.Thread(
        target=ai_worker_loop,
        args=(Config,),
        daemon=True
    )
    ai_thread.start()
    
    try:
        # Start video capture (main thread)
        video_capture_loop(Config)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    finally:
        # Cleanup
        print("\n🔄 Shutting down...")
        stop_event.set()
        ai_thread.join(timeout=5)
        print("✅ Shutdown complete\n")


if __name__ == "__main__":
    main()
