
from core import FrameBuffer, MotionDetector, AIInvestigator, MetricsLogger
from models import EdgeVision, CloudAI
from config import Config
import os
import sys
import cv2
import time
import threading
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


# Global shared resources
frame_buffer = FrameBuffer()
stop_event = threading.Event()
models_ready = threading.Event()  # Signal when AI models are loaded
metrics_logger = MetricsLogger()


def ai_worker_loop(config: Config):
    """
    Background thread that monitors for motion and triggers investigation.
    Uses REAL-TIME frame access for each Cloud question.
    """
    print("\n🔧 [AI THREAD] Initializing models...")

    # Initialize A2A client FIRST if networked mode is enabled
    # This avoids loading local models when remote agents are already running
    a2a_client = None
    if config.ENABLE_A2A:
        try:
            from a2a import A2AClient
            rag_url = config.RAG_AGENT_URL if config.ENABLE_RAG else None
            a2a_client = A2AClient(
                edge_url=config.EDGE_AGENT_URL,
                cloud_url=config.CLOUD_AGENT_URL,
                rag_url=rag_url,
            )
            print(f"✅ [A2A] Networked agent communication enabled")
            print(f"   Edge:  {config.EDGE_AGENT_URL}")
            print(f"   Cloud: {config.CLOUD_AGENT_URL}")
            if rag_url:
                print(f"   RAG:   {config.RAG_AGENT_URL}")

            # Wait for all agents to be reachable before proceeding
            print(f"\n⏳ [A2A] Waiting for agent services to be ready...")
            try:
                a2a_client.wait_for_agents(timeout=60.0)
            except RuntimeError as e:
                print(f"❌ {e}")
                print(f"   Falling back to direct model calls")
                a2a_client = None
        except Exception as e:
            print(f"⚠️  [A2A] Could not initialize: {e}")
            print(f"   Falling back to direct model calls")

    # Initialize AI models — skip if A2A is active (models run as remote services)
    if a2a_client is not None:
        print("📡 [AI THREAD] A2A mode — using remote agents, skipping local model loading")
        edge_vision = None
        cloud_ai = None
    else:
        print("⏳ [AI THREAD] Please wait, this may take a few minutes on first run...")
        edge_vision = EdgeVision(
            model_id=config.EDGE_MODEL_ID,
            device=config.DEVICE
        )
        cloud_ai = CloudAI(
            api_key=config.GROQ_API_KEY,
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

    # Create investigator with metrics
    investigator = AIInvestigator(
        edge_vision=edge_vision,
        cloud_ai=cloud_ai,
        frame_provider=lambda: frame_buffer.get_current_frame(),
        max_rounds=config.MAX_INVESTIGATION_ROUNDS,
        fps=fps,
        save_alerts=config.SAVE_ALERTS,
        alerts_dir=config.ALERTS_DIR,
        buffer_duration=config.BUFFER_DURATION_SECONDS,
        fcm_notifier=fcm_notifier,
        metrics_logger=metrics_logger,
        a2a_client=a2a_client,
    )
    investigator._video_file = Path(config.VIDEO_PATH).name

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

        # Update shared frame buffer with raw BGR (lazy PIL conversion on read)
        frame_buffer.update_frame(frame_bgr, timestamp, motion=motion)

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

        # Print and export benchmark metrics
        metrics_logger.print_summary()
        metrics_logger.export_json("alerts/benchmark_metrics.json")
        metrics_logger.export_csv("alerts/benchmark_metrics.csv")

        print("✅ Shutdown complete\n")


if __name__ == "__main__":
    main()
