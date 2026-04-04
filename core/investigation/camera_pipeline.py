"""Per-Camera Pipeline — FPC-ARC (Freshness-Priority Coalescing with Adaptive Rate Control).

Each CameraPipeline runs its own thread, with a single SLOT (not a queue)
for pending motion events. New events atomically overwrite old ones,
ensuring we always investigate the LATEST scene — not stale data.

Key Algorithm: Slot Pattern + Adaptive Cooldown
  - Queue growth: O(1) per camera (bounded)
  - Events overwrite instead of stack → no stale investigations
  - Cooldown adapts: CLEAR=10s, REVIEW=3s, ALERT=1s (AIMD-inspired)

Architecture:
    CameraSource (capture thread)
        │  motion event → slot.overwrite(event)  [O(1)]
        ▼
    CameraPipeline (investigation thread)
        ├── submit_encode/answer → InferenceServer (shared GPU queue)
        └── assess_threat → CloudAI (independent network I/O)
"""

import cv2
import time
import threading
import traceback
from pathlib import Path
from typing import Optional, Dict, Callable

import numpy as np
from PIL import Image

from .investigator import AIInvestigator
from .escalation_tracker import EscalationTracker


# ── Adaptive Cooldown Thresholds ─────────────────────────────────────

ADAPTIVE_COOLDOWNS = {
    "CLEAR_HIGH":  10.0,   # CLEAR with confidence ≥ 90%: scene is safe
    "CLEAR_LOW":    6.0,   # CLEAR with confidence < 90%: somewhat uncertain
    "REVIEW":       3.0,   # REVIEW: needs attention, stay alert
    "INVESTIGATE":  3.0,   # Unresolved INVESTIGATE: monitor closely
    "ALERT":        1.0,   # ALERT: active threat, keep watching
}

MAX_EVENT_AGE_SECONDS = 5.0  # Drop events older than this (stale frames)


class CameraPipeline:
    """Independent per-camera investigation pipeline using Slot Pattern.

    Instead of a FIFO queue, each camera has a single atomic SLOT:
    - New motion events OVERWRITE the pending slot (O(1), no growth)
    - Worker always processes the FRESHEST event
    - At investigation start, grabs the LATEST frame from camera buffer

    Adaptive cooldown adjusts per-camera rate based on last verdict:
    - CLEAR → relax (10s), ALERT → stay vigilant (1s)
    """

    def __init__(
        self,
        camera_id: str,
        inference_server,
        cloud_ai,
        frame_provider: Callable,
        max_rounds: int = 3,
        fps: float = 30.0,
        initial_edge_max_tokens: int = 24,
        followup_edge_max_tokens: int = 16,
        save_alerts: bool = True,
        alerts_dir: Path = Path("alerts"),
        buffer_duration: int = 10,
        fcm_notifier=None,
        metrics_logger=None,
        a2a_client=None,
        video_file: str = "",
        escalation_tracker: Optional[EscalationTracker] = None,
    ):
        self.camera_id = camera_id
        self.inference_server = inference_server
        self.frame_provider = frame_provider

        # ── Slot Pattern: replaces Queue ────────────────────────────
        self._pending_slot: Optional[Dict] = None
        self._slot_lock = threading.Lock()
        self._slot_event = threading.Event()  # Wake worker on new event

        # ── Adaptive Cooldown state ─────────────────────────────────
        self._last_verdict_time: float = 0.0
        self._current_cooldown: float = 2.0  # Initial cooldown before first verdict
        self._investigation_count: int = 0

        # ── Worker state ────────────────────────────────────────────
        self._worker: Optional[threading.Thread] = None
        self._running = False
        self._stop_event = threading.Event()
        self._active_investigation = False

        # Temporal escalation tracker (shared across cameras or per-camera)
        self._escalation_tracker = escalation_tracker or EscalationTracker()

        # ── Per-camera investigator ─────────────────────────────────
        self.investigator = AIInvestigator(
            edge_vision=None,
            cloud_ai=cloud_ai,
            frame_provider=frame_provider,
            max_rounds=max_rounds,
            fps=fps,
            initial_edge_max_tokens=initial_edge_max_tokens,
            followup_edge_max_tokens=followup_edge_max_tokens,
            save_alerts=save_alerts,
            alerts_dir=alerts_dir / camera_id,
            buffer_duration=buffer_duration,
            fcm_notifier=fcm_notifier,
            metrics_logger=metrics_logger,
            a2a_client=a2a_client,
            inference_server=inference_server,
            camera_id=camera_id,
        )
        self.investigator._video_file = video_file
        self.investigator._camera_id = camera_id

    def start(self, stop_event: threading.Event) -> None:
        """Start the pipeline's investigation worker thread."""
        if self._running:
            return
        self._running = True
        self._stop_event = stop_event
        self._worker = threading.Thread(
            target=self._investigation_loop,
            daemon=True,
            name=f"pipeline-{self.camera_id}",
        )
        self._worker.start()
        print(f"[PIPELINE {self.camera_id}] Worker started (FPC-ARC: slot + adaptive cooldown)")

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the pipeline to stop and wait for active investigation."""
        self._running = False
        self._slot_event.set()  # Wake worker to check stop condition
        if self._worker:
            self._worker.join(timeout=timeout)

    # ── Slot Pattern: Atomic Overwrite ───────────────────────────────

    def submit_motion(
        self,
        timestamp: str,
        trigger_frame_bgr: np.ndarray,
    ) -> bool:
        """Submit a motion event — OVERWRITES any pending event.

        Unlike a queue, this never grows. The newest event always wins.
        The worker wakes up and processes only the latest submission.

        Returns:
            True if accepted, False if a cooldown is active.
        """
        # Adaptive cooldown gate
        if self._is_in_cooldown():
            return False

        event = {
            "timestamp": timestamp,
            "trigger_frame_bgr": trigger_frame_bgr.copy(),
            "enqueued_at": time.time(),
        }

        with self._slot_lock:
            old = self._pending_slot
            self._pending_slot = event  # Atomic overwrite

        if old is not None:
            old_age = time.time() - old["enqueued_at"]
            # Only log if the replaced event was meaningfully old (>=1s)
            # Suppresses ~100 zero-age same-timestamp lines per investigation
            if old_age >= 1.0:
                print(
                    f"[{self.camera_id}] Slot: replaced stale event "
                    f"{old['timestamp']} (age {old_age:.1f}s) with {timestamp}"
                )

        self._slot_event.set()  # Wake the worker
        return True

    @property
    def is_busy(self) -> bool:
        """Whether the pipeline is actively investigating."""
        return self._active_investigation

    @property
    def pending_count(self) -> int:
        """0 or 1 — how many events are waiting (slot pattern)."""
        with self._slot_lock:
            return 1 if self._pending_slot is not None else 0

    # ── Adaptive Cooldown ────────────────────────────────────────────

    def _is_in_cooldown(self) -> bool:
        """Check if this camera is still in adaptive cooldown."""
        if self._last_verdict_time == 0:
            return False  # No verdict yet
        elapsed = time.time() - self._last_verdict_time
        return elapsed < self._current_cooldown

    def _update_cooldown(self, verdict: Dict) -> None:
        """Adapt cooldown based on investigation verdict (AIMD-inspired).

        CLEAR (high) → long cooldown (scene is stable)
        REVIEW       → moderate cooldown (needs attention)
        ALERT        → minimal cooldown (keep watching)
        """
        status = verdict.get("status", "CLEAR")
        confidence = verdict.get("confidence", 0)

        if status == "ALERT":
            self._current_cooldown = ADAPTIVE_COOLDOWNS["ALERT"]
        elif status == "REVIEW":
            self._current_cooldown = ADAPTIVE_COOLDOWNS["REVIEW"]
        elif status == "INVESTIGATE":
            self._current_cooldown = ADAPTIVE_COOLDOWNS["INVESTIGATE"]
        elif confidence >= 90:
            self._current_cooldown = ADAPTIVE_COOLDOWNS["CLEAR_HIGH"]
        else:
            self._current_cooldown = ADAPTIVE_COOLDOWNS["CLEAR_LOW"]

        self._last_verdict_time = time.time()

    # ── Investigation Loop ───────────────────────────────────────────

    def _consume_slot(self) -> Optional[Dict]:
        """Atomically consume the pending slot. Returns event or None."""
        with self._slot_lock:
            event = self._pending_slot
            self._pending_slot = None
            return event

    def _investigation_loop(self) -> None:
        """Main loop: wait for slot events, investigate the freshest frame.

        Algorithm:
            1. Wait for slot_event (no busy-waiting)
            2. Consume the slot (atomic, O(1))
            3. Freshness check — drop if age > MAX_EVENT_AGE_SECONDS
            4. Grab LATEST frame from camera buffer (not the trigger frame)
            5. Run investigation
            6. Update adaptive cooldown based on verdict
        """
        while self._running and not self._stop_event.is_set():
            # Wait for a motion event (timeout allows checking stop condition)
            self._slot_event.wait(timeout=0.5)
            self._slot_event.clear()

            if not self._running or self._stop_event.is_set():
                break

            event = self._consume_slot()
            if event is None:
                continue

            try:
                timestamp = event["timestamp"]
                trigger_bgr = event["trigger_frame_bgr"]
                enqueued_at = event["enqueued_at"]

                # ── Freshness gate ──────────────────────────────
                age = time.time() - enqueued_at
                if age > MAX_EVENT_AGE_SECONDS:
                    print(
                        f"[{self.camera_id}] Dropping stale event "
                        f"({timestamp}, age={age:.1f}s > {MAX_EVENT_AGE_SECONDS}s)"
                    )
                    continue

                queue_wait_ms = max(age * 1000, 0.0)
                self._investigation_count += 1
                inv_id = self._investigation_count

                # ── Grab FRESHEST frame from camera buffer ──────
                # The trigger frame tells us WHEN to look, but we
                # analyze the CURRENT frame for maximum freshness.
                fresh_frame, fresh_ts = self.frame_provider()
                if fresh_frame is not None:
                    frame_to_analyze = fresh_frame
                    analysis_ts = fresh_ts or timestamp
                else:
                    # Fallback to trigger frame if buffer is empty
                    frame_rgb = cv2.cvtColor(trigger_bgr, cv2.COLOR_BGR2RGB)
                    frame_to_analyze = Image.fromarray(frame_rgb)
                    analysis_ts = timestamp

                print(f"\n{'#'*60}")
                print(
                    f"[{self.camera_id}:inv-{inv_id}] MOTION at {timestamp} "
                    f"(wait={queue_wait_ms:.0f}ms, cooldown={self._current_cooldown:.1f}s)"
                )
                print(f"{'#'*60}")

                self._active_investigation = True

                result = self.investigator.investigate_realtime(
                    frame_to_analyze,
                    analysis_ts,
                    queue_wait_ms=queue_wait_ms,
                    queue_depth=self.pending_count,
                )

                self._active_investigation = False

                # ── Update adaptive cooldown ────────────────────
                self._update_cooldown(result)

                status = result.get('status', '?')
                confidence = result.get('confidence', 0)

                # ── Temporal escalation check ───────────────────
                escalation = self._escalation_tracker.add_verdict(
                    camera_id=self.camera_id,
                    status=status,
                    confidence=confidence,
                    inv_id=f"inv-{inv_id}",
                )
                if escalation:
                    # Escalation IS the ALERT — this is the core design:
                    # edge + cloud couldn't resolve it individually, but
                    # persistent suspicious activity = confirmed threat.
                    esc_confidence = escalation["confidence"]
                    esc_reason = escalation["reason"]

                    print(f"\n{'='*60}")
                    print(
                        f"🚨🚨🚨 [{self.camera_id}] [VERDICT] "
                        f"ESCALATION ALERT — PERSISTENT THREAT DETECTED 🚨🚨🚨"
                    )
                    print(f"   Confidence: {esc_confidence}%")
                    print(f"   Reason: {esc_reason}")
                    print(f"{'='*60}\n")

                    # Save as alert (not review)
                    esc_verdict = {
                        "status": "ALERT",
                        "confidence": esc_confidence,
                        "reason": esc_reason,
                        "source": "temporal_escalation",
                    }
                    history = result.get("history", [])
                    self.investigator._save_incident(
                        esc_verdict, history, prefix="escalation_alert",
                    )

                    # FCM notification
                    if self.investigator.fcm_notifier:
                        ts = result.get("timestamp", "unknown")
                        self.investigator.fcm_notifier.send_alert(
                            verdict=esc_verdict,
                            history=history,
                            timestamp=ts,
                            alert_dir=None,
                        )

                    # Override cooldown to ALERT level (keep watching)
                    self._current_cooldown = ADAPTIVE_COOLDOWNS["ALERT"]
                    self._last_verdict_time = time.time()
                    status = "ALERT (escalated)"
                    confidence = esc_confidence

                print(
                    f"[{self.camera_id}:inv-{inv_id}] "
                    f"Next cooldown: {self._current_cooldown:.1f}s "
                    f"(based on {status})"
                )

            except Exception as e:
                self._active_investigation = False
                print(f"[PIPELINE {self.camera_id}] Investigation failed: {e}")
                traceback.print_exc()
