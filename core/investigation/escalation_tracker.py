"""Temporal Escalation Engine — cross-investigation memory for each camera.

Problem it solves:
    Each investigation is stateless. A bar fight that produces 7 REVIEW verdicts
    in 80 seconds never triggers ALERT because no single investigation sees the
    full context.

Algorithm:
    - Rolling verdict window per camera (time-bounded)
    - If >= THRESHOLD consecutive non-CLEAR verdicts in WINDOW_SEC -> ESCALATE
    - CLEAR with high confidence (>= 85%) resets the counter for that camera
    - Hard cooldown between escalations prevents alert storms
"""

import time
import threading
from collections import deque
from typing import Optional, Dict, List
from dataclasses import dataclass, field


# --- Escalation parameters ---
ESCALATION_THRESHOLD = 2       # consecutive suspicious verdicts to trigger escalation
ESCALATION_WINDOW_SEC = 120.0  # rolling time window for counting verdicts
ESCALATION_COOLDOWN_SEC = 30.0 # min time between escalation alerts from same camera
CLEAR_RESET_CONFIDENCE = 80    # CLEAR with confidence >= this resets the counter


@dataclass
class VerdictEntry:
    status: str            # CLEAR / INVESTIGATE / REVIEW / ALERT
    confidence: int
    timestamp: float       # wall clock time (time.time())
    investigation_id: str  # e.g. "inv-4" for logging


@dataclass
class CameraEscalationState:
    verdicts: deque = field(default_factory=lambda: deque(maxlen=50))
    last_escalation_time: Optional[float] = None
    consecutive_non_clear: int = 0


class EscalationTracker:
    """Thread-safe per-camera verdict tracker with temporal escalation.

    Usage in CameraPipeline:
        tracker = EscalationTracker()

        # After each investigation verdict:
        escalation = tracker.add_verdict(
            camera_id="cam0",
            status="REVIEW",
            confidence=80,
            inv_id="inv-4",
        )
        if escalation:
            # Trigger alert handlers (FCM, save, log)
            handle_escalation(escalation)
    """

    def __init__(self):
        self._states: Dict[str, CameraEscalationState] = {}
        self._lock = threading.Lock()

    def _get_state(self, camera_id: str) -> CameraEscalationState:
        if camera_id not in self._states:
            self._states[camera_id] = CameraEscalationState()
        return self._states[camera_id]

    def add_verdict(
        self,
        camera_id: str,
        status: str,
        confidence: int,
        inv_id: str = "",
    ) -> Optional[Dict]:
        """Record a verdict and check for escalation.

        Returns:
            Escalation alert dict if escalation triggered, else None.
        """
        now = time.time()

        with self._lock:
            state = self._get_state(camera_id)

            entry = VerdictEntry(
                status=status,
                confidence=confidence,
                timestamp=now,
                investigation_id=inv_id,
            )
            state.verdicts.append(entry)

            # --- 1. CLEAR with high confidence resets counter ---
            if status == "CLEAR" and confidence >= CLEAR_RESET_CONFIDENCE:
                if state.consecutive_non_clear > 0:
                    print(
                        f"[ESCALATION {camera_id}] CLEAR ({confidence}%) resets "
                        f"counter (was {state.consecutive_non_clear} consecutive)"
                    )
                state.consecutive_non_clear = 0
                return None

            # --- 2. Non-CLEAR verdict (or low-confidence CLEAR): increment ---
            state.consecutive_non_clear += 1

            # --- 3. Collect recent suspicious verdicts for context ---
            # "Suspicious" = anything that ISN'T a high-confidence CLEAR
            # This matches the counter logic: low-confidence CLEARs
            # (e.g. timeout fallbacks at 50%) count as suspicious.
            cutoff = now - ESCALATION_WINDOW_SEC
            recent_suspicious: List[VerdictEntry] = [
                v for v in state.verdicts
                if v.timestamp >= cutoff
                and not (v.status == "CLEAR" and v.confidence >= CLEAR_RESET_CONFIDENCE)
            ]

            # --- 4. Check escalation threshold (use COUNTER, not filter) ---
            # The counter is authoritative — it correctly tracks the
            # consecutive suspicious sequence including timeout fallbacks.
            if state.consecutive_non_clear < ESCALATION_THRESHOLD:
                label = status if status != "CLEAR" else f"CLEAR({confidence}%)"
                print(
                    f"   [ESCALATION {camera_id}] "
                    f"{state.consecutive_non_clear} consecutive suspicious ({label}) "
                    f"(need {ESCALATION_THRESHOLD} to escalate)"
                )
                return None

            # --- 5. Cooldown check ---
            if (
                state.last_escalation_time is not None
                and now - state.last_escalation_time < ESCALATION_COOLDOWN_SEC
            ):
                remaining = ESCALATION_COOLDOWN_SEC - (now - state.last_escalation_time)
                print(
                    f"   [ESCALATION {camera_id}] Threshold met but in cooldown "
                    f"({remaining:.0f}s remaining)"
                )
                return None

            # --- 6. ESCALATE ---
            state.last_escalation_time = now
            state.consecutive_non_clear = 0  # reset after escalation

            # Build reason from all suspicious verdicts in window
            suspicious_count = len(recent_suspicious)
            window_secs = (
                now - recent_suspicious[0].timestamp
                if recent_suspicious else 0.0
            )
            max_conf = (
                max(v.confidence for v in recent_suspicious)
                if recent_suspicious else confidence
            )
            first_inv = (
                recent_suspicious[0].investigation_id
                if recent_suspicious else inv_id
            )

            reason = (
                f"Persistent incident: Camera {camera_id} flagged "
                f"{suspicious_count} suspicious verdicts "
                f"in {window_secs:.0f}s "
                f"(inv {first_inv} -> {inv_id}). "
                f"Repeated distress signals — requires immediate inspection."
            )

            print(
                f"\n{'#'*60}\n"
                f"[ESCALATION {camera_id}] 🚨 TEMPORAL ESCALATION TRIGGERED\n"
                f"   Suspicious verdicts: {suspicious_count} in {window_secs:.0f}s\n"
                f"   Max confidence seen: {max_conf}%\n"
                f"   Reason: {reason}\n"
                f"{'#'*60}\n"
            )

            return {
                "status": "ALERT",
                "confidence": min(max_conf + 10, 95),
                "reason": reason,
                "source": "escalation",
                "camera_id": camera_id,
                "verdicts_count": suspicious_count,
                "window_seconds": window_secs,
            }

    def reset(self, camera_id: str) -> None:
        """Forcefully reset the escalation counter for a camera."""
        with self._lock:
            if camera_id in self._states:
                self._states[camera_id].consecutive_non_clear = 0

    def get_state_summary(self, camera_id: str) -> Dict:
        """Return current escalation state for logging/display."""
        with self._lock:
            if camera_id not in self._states:
                return {"consecutive_non_clear": 0, "verdict_count": 0}
            state = self._states[camera_id]
            return {
                "consecutive_non_clear": state.consecutive_non_clear,
                "verdict_count": len(state.verdicts),
                "last_escalation_ago": (
                    time.time() - state.last_escalation_time
                    if state.last_escalation_time
                    else None
                ),
            }
