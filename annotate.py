"""Ground Truth Annotation Tool for Violence Detection Benchmarks.

Play a video and mark VIOLENCE / NO_VIOLENCE segments with keypresses.
Outputs JSON compatible with benchmark.py and core/evaluation.py.

Usage:
    python annotate.py path/to/video.mp4
    python annotate.py path/to/video.mp4 --start-label VIOLENCE

Controls:
    v         Mark VIOLENCE starts here
    n         Mark NO_VIOLENCE starts here
    Space     Pause / Resume
    Right     Seek +5 seconds
    Left      Seek -5 seconds
    .         (Paused) Step forward 1 frame
    ,         (Paused) Step backward 1 frame
    + / =     Increase playback speed
    -         Decrease playback speed
    u         Undo last transition
    q         Save and quit
    Esc       Quit without saving
"""

import cv2
import numpy as np
import json
import argparse
import os
import sys

SPEED_STEPS = [0.25, 0.5, 1.0, 1.5, 2.0, 4.0]
SEEK_SECONDS = 5

# Colors (BGR)
RED = (0, 0, 200)
GREEN = (0, 160, 0)
WHITE = (255, 255, 255)
LIGHT_GRAY = (200, 200, 200)
CYAN = (255, 255, 0)
BLACK = (0, 0, 0)

# Arrow key codes (Linux / macOS / Windows variants)
LEFT_ARROWS = {65361, 0x250000, 63234, 2424832}
RIGHT_ARROWS = {65363, 0x270000, 63235, 2555904}


def seconds_to_hhmmss(sec):
    sec = max(0, sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class AnnotationPlayer:
    def __init__(self, video_path, start_label="NO_VIOLENCE"):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print(f"Error: cannot open video: {video_path}")
            sys.exit(1)

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_sec = self.total_frames / self.fps if self.fps > 0 else 0
        self.current_frame = 0

        # Playback state
        self.paused = False
        self.speed_idx = SPEED_STEPS.index(1.0)
        self.should_quit = False
        self.save_on_quit = False

        # Transitions: sorted list of (frame_number, label)
        self.transitions = [(0, start_label)]
        # Undo stack: records transitions in the order the user added them
        self.undo_stack = []

        # Cache for paused frame
        self._paused_frame = None

    @property
    def speed(self):
        return SPEED_STEPS[self.speed_idx]

    def get_current_label(self):
        label = self.transitions[0][1]
        for frame_num, lbl in self.transitions:
            if frame_num <= self.current_frame:
                label = lbl
            else:
                break
        return label

    def add_transition(self, label):
        if self.get_current_label() == label:
            return
        # Remove any existing transition at this exact frame (except frame 0 initial)
        self.transitions = [
            (f, l) for f, l in self.transitions
            if f != self.current_frame or f == 0
        ]
        if self.current_frame == 0:
            # Replace the initial label
            self.transitions[0] = (0, label)
        else:
            self.transitions.append((self.current_frame, label))
            self.transitions.sort(key=lambda t: t[0])
        self.undo_stack.append(self.current_frame)

    def undo_last_transition(self):
        if not self.undo_stack:
            return
        frame_to_remove = self.undo_stack.pop()
        if frame_to_remove == 0:
            # Revert initial label to NO_VIOLENCE
            self.transitions[0] = (0, "NO_VIOLENCE")
        else:
            self.transitions = [(f, l) for f, l in self.transitions if f != frame_to_remove]
        if not self.transitions:
            self.transitions = [(0, "NO_VIOLENCE")]

    def seek_relative(self, seconds):
        target = self.current_frame + int(seconds * self.fps)
        target = max(0, min(target, self.total_frames - 1))
        self.current_frame = target
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        self._paused_frame = None

    def draw_overlay(self, frame):
        h, w = frame.shape[:2]

        # --- Top bar (semi-transparent) ---
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 44), BLACK, -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        cur_sec = self.current_frame / self.fps if self.fps > 0 else 0
        ts = f"{seconds_to_hhmmss(cur_sec)} / {seconds_to_hhmmss(self.duration_sec)}"
        cv2.putText(frame, ts, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)

        frame_text = f"Frame: {self.current_frame}/{self.total_frames}"
        cv2.putText(frame, frame_text, (w // 3, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, LIGHT_GRAY, 1)

        speed_text = f"Speed: {self.speed}x"
        tw = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
        cv2.putText(frame, speed_text, (w - tw - 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, LIGHT_GRAY, 1)

        if self.paused:
            cv2.putText(frame, "|| PAUSED", (w // 2 - 50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, CYAN, 2)

        # --- Timeline bar ---
        tl_y = h - 75
        tl_h = 16
        margin = 10
        bar_w = w - 2 * margin

        # Draw segment colors
        for i, (tf, label) in enumerate(self.transitions):
            x0 = margin + int(tf / max(self.total_frames, 1) * bar_w)
            if i + 1 < len(self.transitions):
                x1 = margin + int(self.transitions[i + 1][0] / max(self.total_frames, 1) * bar_w)
            else:
                x1 = margin + bar_w
            color = RED if label == "VIOLENCE" else GREEN
            cv2.rectangle(frame, (x0, tl_y), (x1, tl_y + tl_h), color, -1)

        # Transition markers (small white ticks)
        for tf, _ in self.transitions:
            if tf == 0:
                continue
            tx = margin + int(tf / max(self.total_frames, 1) * bar_w)
            cv2.line(frame, (tx, tl_y - 4), (tx, tl_y + tl_h + 4), WHITE, 1)

        # Playback cursor
        cx = margin + int(self.current_frame / max(self.total_frames, 1) * bar_w)
        cv2.line(frame, (cx, tl_y - 5), (cx, tl_y + tl_h + 5), WHITE, 2)

        # Border
        cv2.rectangle(frame, (margin, tl_y), (margin + bar_w, tl_y + tl_h), WHITE, 1)

        # --- Bottom status bar ---
        current_label = self.get_current_label()
        bar_color = RED if current_label == "VIOLENCE" else GREEN
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, h - 50), (w, h), bar_color, -1)
        cv2.addWeighted(overlay2, 0.8, frame, 0.2, 0, frame)

        cv2.putText(frame, f"[ {current_label} ]", (10, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, WHITE, 2)

        hints = "v:Violence  n:Normal  Space:Pause  u:Undo  q:Save"
        cv2.putText(frame, hints, (w // 3, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE, 1)

        # Segment count
        seg_text = f"Segments: {len(self.transitions)}"
        tw2 = cv2.getTextSize(seg_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
        cv2.putText(frame, seg_text, (w - tw2 - 10, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)

    def handle_key(self, key):
        if key == -1:
            return
        char = key & 0xFF

        if char == ord('q'):
            self.should_quit = True
            self.save_on_quit = True
        elif key == 27:  # Esc
            self.should_quit = True
            self.save_on_quit = False
        elif char == ord(' '):
            self.paused = not self.paused
            self._paused_frame = None
        elif char == ord('v'):
            self.add_transition("VIOLENCE")
        elif char == ord('n'):
            self.add_transition("NO_VIOLENCE")
        elif key in RIGHT_ARROWS:
            self.seek_relative(SEEK_SECONDS)
        elif key in LEFT_ARROWS:
            self.seek_relative(-SEEK_SECONDS)
        elif char == ord('.') and self.paused:
            self.current_frame = min(self.current_frame + 1, self.total_frames - 1)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            self._paused_frame = None
        elif char == ord(',') and self.paused:
            self.current_frame = max(self.current_frame - 1, 0)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            self._paused_frame = None
        elif char in (ord('+'), ord('=')):
            self.speed_idx = min(self.speed_idx + 1, len(SPEED_STEPS) - 1)
        elif char == ord('-'):
            self.speed_idx = max(self.speed_idx - 1, 0)
        elif char == ord('u'):
            self.undo_last_transition()

    def build_segments(self):
        segments = []
        for i, (frame, label) in enumerate(self.transitions):
            start_sec = frame / self.fps
            if i + 1 < len(self.transitions):
                end_sec = (self.transitions[i + 1][0] - 1) / self.fps
            else:
                end_sec = (self.total_frames - 1) / self.fps
            # Merge with previous if same label
            if segments and segments[-1]["label"] == label:
                segments[-1]["end_time"] = seconds_to_hhmmss(end_sec)
                continue
            segments.append({
                "id": len(segments) + 1,
                "start_time": seconds_to_hhmmss(start_sec),
                "end_time": seconds_to_hhmmss(end_sec),
                "label": label,
                "description": ""
            })
        return segments

    def save_annotations(self):
        segments = self.build_segments()

        # Relative video path
        project_root = os.path.dirname(os.path.abspath(__file__))
        video_abs = os.path.abspath(self.video_path)
        try:
            video_rel = os.path.relpath(video_abs, project_root)
        except ValueError:
            video_rel = self.video_path

        output_data = {
            "video_file": video_rel,
            "annotations": segments
        }

        video_stem = os.path.splitext(os.path.basename(self.video_path))[0]
        output_dir = os.path.join(project_root, "annotations")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{video_stem}.json")

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\nSaved: {output_path}")
        print(f"  {len(segments)} segment(s):")
        for s in segments:
            print(f"    {s['start_time']} - {s['end_time']}: {s['label']}")

    def run(self):
        win = "Annotate - Ground Truth"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 960, 540)

        print(f"\nPlaying: {self.video_path}")
        print(f"  FPS: {self.fps:.1f} | Frames: {self.total_frames} | Duration: {seconds_to_hhmmss(self.duration_sec)}")
        print(f"  Starting label: {self.transitions[0][1]}")
        print(f"\nControls: v=Violence  n=Normal  Space=Pause  Arrows=Seek  u=Undo  q=Save+Quit  Esc=Quit\n")

        while self.cap.isOpened():
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    # End of video - auto save
                    self.save_on_quit = True
                    break
                self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                self._paused_frame = frame.copy()
            else:
                if self._paused_frame is None:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    self._paused_frame = frame.copy()
                frame = self._paused_frame.copy()

            self.draw_overlay(frame)
            cv2.imshow(win, frame)

            delay = max(1, int(1000 / (self.fps * self.speed)))
            if self.paused:
                delay = 50

            key = cv2.waitKeyEx(delay)
            self.handle_key(key)

            if self.should_quit:
                break

        self.cap.release()
        cv2.destroyAllWindows()

        if self.save_on_quit:
            self.save_annotations()
        else:
            print("\nQuit without saving.")


def main():
    parser = argparse.ArgumentParser(
        description="Annotate ground truth for violence detection benchmarks.",
        epilog="Output: annotations/<video_name>.json"
    )
    parser.add_argument("video", help="Path to video file")
    parser.add_argument(
        "--start-label", default="NO_VIOLENCE",
        choices=["VIOLENCE", "NO_VIOLENCE"],
        help="Label at the start of the video (default: NO_VIOLENCE)"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        print(f"Error: file not found: {args.video}")
        sys.exit(1)

    player = AnnotationPlayer(args.video, args.start_label)
    player.run()


if __name__ == "__main__":
    main()
