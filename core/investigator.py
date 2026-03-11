"""AI Investigator - Real-Time Threat Analysis with Pipeline Parallelism."""

import time
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime
from typing import Callable, Tuple, Optional, Dict, List
from PIL import Image


class AIInvestigator:
    """
    Real-time AI Investigator with threat-triggered conversation loop.

    Flow:
    1. Motion detected -> Edge analyzes FIRST frame -> sends to Cloud
    2. Cloud says CLEAR -> back to monitoring
    3. Cloud says INVESTIGATE -> Edge grabs CURRENT frame -> analyzes -> sends to Cloud
    4. Loop continues until Cloud says CLEAR or ALERT

    Pipeline Optimization:
    While Cloud processes round N, we pre-encode the next frame on the GPU.
    When Cloud returns with round N+1's question, we only need the lightweight
    answer() step instead of the full encode+answer cycle. This hides 200-500ms
    of GPU encoding time behind Cloud API latency.
    """

    def __init__(self, edge_vision, cloud_ai, frame_provider: Callable,
                 max_rounds: int = 3, fps: float = 30.0,
                 save_alerts: bool = True, alerts_dir: Path = Path("alerts"),
                 buffer_duration: int = 10, fcm_notifier=None,
                 metrics_logger=None, a2a_client=None):
        """
        Initialize AI investigator.

        Args:
            edge_vision: EdgeVision instance (used for direct calls)
            cloud_ai: CloudAI instance (used for direct calls)
            frame_provider: Callback function that returns (pil_image, timestamp)
            max_rounds: Maximum investigation rounds to prevent infinite loops
            fps: Video frames per second (for buffer size calculation)
            save_alerts: Whether to save frames on ALERT verdict
            alerts_dir: Directory to save alert incidents
            buffer_duration: How many seconds of frames to buffer
            fcm_notifier: Optional FCMNotifier instance for push notifications
            metrics_logger: Optional MetricsLogger for performance benchmarking
            a2a_client: Optional A2AClient for networked agent communication.
                        When provided, uses HTTP calls to Edge/Cloud/RAG agents
                        instead of direct method calls.
        """
        self.edge = edge_vision
        self.cloud = cloud_ai
        self.get_frame = frame_provider
        self.max_rounds = max_rounds
        self.fcm_notifier = fcm_notifier
        self.metrics = metrics_logger
        self.a2a = a2a_client

        # Alert saving configuration
        self.save_alerts = save_alerts
        self.alerts_dir = Path(alerts_dir)

        if self.save_alerts:
            self.alerts_dir.mkdir(parents=True, exist_ok=True)

        # Circular buffer for frame history
        buffer_size = int(fps * buffer_duration)
        self.frame_buffer = deque(maxlen=buffer_size)

        print("✅ [AI] Violence Detection System Ready")


    def buffer_frame(self, frame: Image.Image, timestamp: str, motion: bool = False):
        """
        Add frame to circular buffer.

        Args:
            frame: PIL Image
            timestamp: Frame timestamp
            motion: Whether motion was detected
        """
        self.frame_buffer.append({
            'frame': frame,
            'timestamp': timestamp,
            'motion': motion
        })

    def investigate_realtime(self, initial_frame: Image.Image,
                           initial_timestamp: str) -> Dict:
        """
        Real-time threat-triggered investigation loop with pipeline parallelism.

        Pipeline optimization: While the cloud API processes round N, we
        pre-encode the next frame on the GPU. This overlaps GPU compute
        with network I/O, saving 200-500ms per round.

        Args:
            initial_frame: First frame that triggered motion
            initial_timestamp: Timestamp of first frame

        Returns:
            Dict with investigation results
        """
        print(f"\n{'='*60}")
        print(f"🔍 [INVESTIGATION] Started at {initial_timestamp}")
        print(f"{'='*60}")

        history: List[str] = []

        # Start metrics tracking
        inv_id = None
        if self.metrics:
            video_file = getattr(self, '_video_file', 'unknown')
            inv_id = self.metrics.start_investigation(video_file, initial_timestamp)
            self.metrics.sample_system_metrics()

        # Buffer the initial frame
        self.buffer_frame(initial_frame, initial_timestamp, motion=True)

        # STEP 1: Initial analysis (Edge) — full encode + answer
        print(f"\n📸 [EDGE @{initial_timestamp}] Analyzing initial frame...")
        start_time = time.time()

        initial_question = "Describe the scene: how many people, their actions, body positions, any physical contact, aggression, distress, or weapons?"
        if self.a2a:
            initial_desc = self.a2a.observe(initial_frame, initial_question)
        else:
            initial_desc = self.edge.analyze(initial_frame, initial_question)
        edge_time = time.time() - start_time

        print(f"   Response: {initial_desc} (⏱️ {edge_time:.2f}s)")
        history.append(f"[{initial_timestamp}] Initial scan: {initial_desc}")

        if self.metrics and inv_id:
            self.metrics.record_edge_call(inv_id, edge_time, initial_desc)

        # RAG retrieval — find similar past cases before cloud assessment
        rag_context = None
        if self.a2a and self.a2a.has_rag:
            rag_context = self.a2a.retrieve(" ".join(history), limit=3)
            if rag_context:
                print(f"   📚 [RAG] Found {len(rag_context)} similar past cases")

        # STEP 2: Cloud assessment + parallel pre-encode of next frame
        print(f"\n☁️  [CLOUD] Analyzing initial report...")
        cloud_start = time.time()

        # Run cloud call and pre-encode next frame in parallel
        pre_encoded = None
        pre_encoded_frame = None
        pre_encoded_ts = None

        with ThreadPoolExecutor(max_workers=2) as executor:
            if self.a2a:
                cloud_future = executor.submit(
                    self.a2a.assess, history, rag_context)
            else:
                cloud_future = executor.submit(
                    self.cloud.assess_threat, history, rag_context)

            # While cloud is thinking, grab and pre-encode the next frame
            next_frame, next_ts = self.get_frame()
            if next_frame is not None:
                try:
                    if self.a2a:
                        pre_encoded = self.a2a.encode(next_frame)
                    else:
                        pre_encoded = self.edge.encode(next_frame)
                    pre_encoded_frame = next_frame
                    pre_encoded_ts = next_ts
                except Exception:
                    pass  # Fall back to full analyze in next round

            decision = cloud_future.result()

        status = decision.get('status', 'CLEAR')
        confidence = decision.get('confidence', 0)
        reason = decision.get('reason', '')

        cloud_time = time.time() - cloud_start

        print(f"   Status: {status} | Confidence: {confidence}% (⏱️ {cloud_time:.2f}s)")
        print(f"   Reason: {reason}")

        if self.metrics and inv_id:
            prompt_len = len("\n".join(history))
            self.metrics.record_cloud_call(inv_id, cloud_time, decision, prompt_len)

        # STEP 3: Investigation loop with pipeline parallelism
        round_num = 1
        while status == "INVESTIGATE" and round_num < self.max_rounds:
            round_num += 1
            round_start = time.time()

            question = decision.get('question', 'Describe any aggressive actions.')
            print(f"\n☁️  [CLOUD] Asking: \"{question}\"")

            # Use pre-encoded frame if available, otherwise grab fresh
            if pre_encoded is not None and pre_encoded_frame is not None:
                current_frame = pre_encoded_frame
                current_ts = pre_encoded_ts
                enc_image = pre_encoded
                print(f"📸 [EDGE @{current_ts}] Answering with pre-encoded frame...")
            else:
                current_frame, current_ts = self.get_frame()
                if current_frame is None:
                    print(f"   ⚠️  [EDGE] No frame available, waiting...")
                    time.sleep(0.5)
                    current_frame, current_ts = self.get_frame()
                    if current_frame is None:
                        print(f"   ❌ [EDGE] Still no frame, aborting investigation")
                        break
                print(f"📸 [EDGE @{current_ts}] Encoding + answering CURRENT frame...")
                if self.a2a:
                    enc_image = self.a2a.encode(current_frame)
                else:
                    enc_image = self.edge.encode(current_frame)

            # Buffer the current frame
            self.buffer_frame(current_frame, current_ts)

            # Phase 2: Answer question using (pre-)encoded image — fast step
            if self.a2a:
                answer = self.a2a.answer(enc_image, question)
            else:
                answer = self.edge.answer(enc_image, question)

            edge_round_time = time.time() - round_start
            print(f"   Response: {answer} (⏱️ {edge_round_time:.2f}s)")

            if self.metrics and inv_id:
                self.metrics.record_edge_call(inv_id, edge_round_time, answer)

            # Add to history
            history.append(f"[{current_ts}] Q: {question}")
            history.append(f"[{current_ts}] A: {answer}")

            # Cloud assessment + parallel pre-encode for potential next round
            print(f"\n☁️  [CLOUD] Analyzing update (Round {round_num})...")
            cloud_start = time.time()

            pre_encoded = None
            pre_encoded_frame = None
            pre_encoded_ts = None

            with ThreadPoolExecutor(max_workers=2) as executor:
                if self.a2a:
                    cloud_future = executor.submit(
                        self.a2a.assess, history, rag_context)
                else:
                    cloud_future = executor.submit(
                        self.cloud.assess_threat, history, rag_context)

                # Pre-encode next frame while cloud is processing
                if round_num < self.max_rounds:
                    next_frame, next_ts = self.get_frame()
                    if next_frame is not None:
                        try:
                            if self.a2a:
                                pre_encoded = self.a2a.encode(next_frame)
                            else:
                                pre_encoded = self.edge.encode(next_frame)
                            pre_encoded_frame = next_frame
                            pre_encoded_ts = next_ts
                        except Exception:
                            pass

                decision = cloud_future.result()

            status = decision.get('status', 'CLEAR')
            confidence = decision.get('confidence', 0)
            reason = decision.get('reason', '')

            cloud_round_time = time.time() - cloud_start
            total_round_time = time.time() - round_start

            print(f"   Status: {status} | Confidence: {confidence}% (⏱️ {cloud_round_time:.2f}s)")
            print(f"   Reason: {reason}")
            print(f"   ⚡ Round {round_num} total: {total_round_time:.2f}s")

            if self.metrics and inv_id:
                prompt_len = len("\n".join(history))
                self.metrics.record_cloud_call(inv_id, cloud_round_time, decision, prompt_len)
                self.metrics.sample_system_metrics()

        # FINAL VERDICT
        print(f"\n{'='*60}")
        if status == "ALERT":
            print(f"🚨🚨🚨 [VERDICT] VIOLENCE DETECTED! 🚨🚨🚨")
            print(f"   Confidence: {confidence}%")
            print(f"   Reason: {reason}")
            print(f"   Rounds of investigation: {round_num}")

            # Save alert frames if enabled
            alert_dir_name = None
            if self.save_alerts and len(self.frame_buffer) > 0:
                from .alert_saver import save_alert_incident
                frames_data = list(self.frame_buffer)
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                alert_dir_name = f"alert_{timestamp}_conf{confidence}"
                save_alert_incident(frames_data, self.alerts_dir, decision, history)

            # Send FCM notification if enabled
            if self.fcm_notifier:
                self.fcm_notifier.send_alert(
                    verdict=decision,
                    history=history,
                    timestamp=initial_timestamp,
                    alert_dir=alert_dir_name
                )

        elif status == "CLEAR":
            print(f"✅ [VERDICT] ALL CLEAR - No threat detected")
            print(f"   Confidence: {confidence}%")
            print(f"   Reason: {reason}")
        elif status == "INVESTIGATE":
            # Safety-first: if we couldn't confirm safe after max rounds, escalate
            status = "ALERT"
            confidence = max(confidence, 80)
            reason = f"{reason} (auto-escalated: unresolved after {self.max_rounds} rounds)"
            print(f"🚨🚨🚨 [VERDICT] AUTO-ESCALATED TO ALERT 🚨🚨🚨")
            print(f"   Could not confirm safe after {self.max_rounds} rounds")
            print(f"   Confidence: {confidence}%")
            print(f"   Reason: {reason}")

            if self.save_alerts and len(self.frame_buffer) > 0:
                from .alert_saver import save_alert_incident
                frames_data = list(self.frame_buffer)
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                alert_dir_name = f"alert_{timestamp}_conf{confidence}"
                save_alert_incident(frames_data, self.alerts_dir, decision, history)

            if self.fcm_notifier:
                self.fcm_notifier.send_alert(
                    verdict={"status": status, "confidence": confidence, "reason": reason},
                    history=history,
                    timestamp=initial_timestamp,
                    alert_dir=alert_dir_name if self.save_alerts else None
                )
        else:
            print(f"⚠️  [VERDICT] Investigation ended unexpectedly")
            print(f"   Last status: {status}")
        print(f"{'='*60}\n")

        # Finalize metrics
        if self.metrics and inv_id:
            self.metrics.end_investigation(inv_id, status, confidence, round_num)
            self.metrics.sample_system_metrics()

        # Auto-ingest into RAG knowledge base (fire-and-forget)
        if self.a2a and self.a2a.has_rag:
            inv_unique_id = f"inv_{initial_timestamp}_{int(time.time())}"
            verdict_dict = {"status": status, "confidence": confidence, "reason": reason}
            threading.Thread(
                target=self.a2a.ingest,
                args=(history, verdict_dict, inv_unique_id),
                kwargs={"metadata": {"timestamp": initial_timestamp, "rounds": round_num}},
                daemon=True,
            ).start()
            print(f"   📚 [RAG] Ingesting investigation into knowledge base...")

        return {
            "status": status,
            "confidence": confidence,
            "reason": reason,
            "history": history,
            "rounds": round_num,
            "timestamp": initial_timestamp
        }
