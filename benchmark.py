"""Headless Benchmark Runner for Violence Detection Evaluation.

Runs the exact same pipeline (EdgeVision, CloudAI, MotionDetector, AIInvestigator)
against annotated test videos and computes a full confusion matrix with standard
classification metrics.

Real-world simulation: After each investigation, the video fast-forwards by the
wall-clock time the investigation took — because in a live system the camera keeps
recording while the AI is busy.  The frame provider also advances through the video
during multi-round investigations so that each round sees a DIFFERENT (later) frame,
just like the real system's FrameBuffer would provide.

Usage:
    python benchmark.py --video videos/test_video.mp4 \
                        --annotations annotations/test_video.json \
                        --output results/benchmark_test_video.json \
                        --investigate-as VIOLENCE
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
from PIL import Image

from config import Config
from core import FrameBuffer, MotionDetector, AIInvestigator, MetricsLogger
from core.evaluation import (
    ConfusionMatrix,
    match_investigation_to_annotation,
    generate_roc_data,
)
from models import EdgeVision, CloudAI


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark violence detection against ground truth annotations"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=Config.VIDEO_PATH,
        help="Path to test video file",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        required=True,
        help="Path to ground truth annotations JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for output JSON report (default: results/benchmark_<video>.json)",
    )
    parser.add_argument(
        "--investigate-as",
        type=str,
        choices=["VIOLENCE", "NO_VIOLENCE", "EXCLUDE"],
        default="VIOLENCE",
        help="How to map INVESTIGATE (timeout) status (default: VIOLENCE)",
    )
    return parser.parse_args()


def load_annotations(path: str) -> dict:
    """Load and validate ground truth annotations from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)

    if "annotations" not in data:
        print(f"Error: annotations file missing 'annotations' key")
        sys.exit(1)

    for ann in data["annotations"]:
        for field in ("start_time", "end_time", "label"):
            if field not in ann:
                print(f"Error: annotation {ann.get('id', '?')} missing '{field}'")
                sys.exit(1)

    print(f"Loaded {len(data['annotations'])} annotations from {path}")
    return data


def run_benchmark(video_path: str, annotations: list, investigate_as: str):
    """Run the full benchmark pipeline simulating real-world timing.

    Real-world behaviour:
    - The camera records continuously at 30fps.
    - When motion triggers an investigation, the AI takes 2-5 seconds.
    - During that time the video keeps playing — the AI doesn't pause the world.
    - After the investigation finishes, the next motion event can only come from
      frames AFTER the video time that elapsed during the investigation.
    - Inside a multi-round investigation the frame_provider returns progressively
      later frames (the camera doesn't freeze while the AI is thinking).

    Args:
        video_path: Path to video file
        annotations: List of ground truth annotation dicts
        investigate_as: How to map INVESTIGATE status

    Returns:
        Tuple of (ConfusionMatrix, MetricsLogger, list of investigation results)
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0

    print(f"\nVideo: {Path(video_path).name}")
    print(f"FPS: {fps:.2f}, Frames: {total_frames}, Duration: {duration_sec:.1f}s")

    # Initialize pipeline components
    print("\nInitializing AI models...")
    edge_vision = EdgeVision(model_id=Config.EDGE_MODEL_ID, device=Config.DEVICE)
    cloud_ai = CloudAI(api_key=Config.GROQ_API_KEY, model_id=Config.CLOUD_MODEL_ID)

    metrics_logger = MetricsLogger()

    # --- Live frame provider ---
    # During an investigation the video time advances by real wall-clock time.
    # The frame_provider reads a LATER frame from the video each time it is
    # called (simulating the real FrameBuffer which always has the latest frame).
    provider_cap = cv2.VideoCapture(video_path)  # separate handle for provider
    provider_base_frame = [0]  # frame number when the investigation started
    provider_wall_start = [0.0]  # wall-clock time when the investigation started

    def frame_provider():
        """Return the frame the camera would be showing RIGHT NOW.

        Uses wall-clock elapsed time since investigation start to compute
        which video frame the camera would have advanced to.
        """
        elapsed = time.time() - provider_wall_start[0]
        target_frame = provider_base_frame[0] + int(elapsed * fps)
        target_frame = min(target_frame, total_frames - 1)

        provider_cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame_bgr = provider_cap.read()
        if not ret:
            return None, None

        secs = target_frame / fps
        ts = time.strftime("%H:%M:%S", time.gmtime(secs))
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        return pil_image, ts

    investigator = AIInvestigator(
        edge_vision=edge_vision,
        cloud_ai=cloud_ai,
        frame_provider=frame_provider,
        max_rounds=Config.MAX_INVESTIGATION_ROUNDS,
        fps=fps,
        save_alerts=False,  # Don't save alert frames during benchmarks
        fcm_notifier=None,  # No notifications during benchmarks
        metrics_logger=metrics_logger,
    )
    investigator._video_file = Path(video_path).name

    motion_detector = MotionDetector(
        threshold=Config.MOTION_THRESHOLD,
        blur_kernel=Config.MOTION_BLUR_KERNEL,
        binary_threshold=Config.MOTION_BINARY_THRESHOLD,
        dilate_iterations=Config.MOTION_DILATE_ITERATIONS,
    )

    # Process video frame-by-frame
    confusion = ConfusionMatrix(investigate_as=investigate_as)
    investigation_results = []
    frame_count = 0

    print(f"\nProcessing video (real-world simulation)...")
    print(f"{'='*60}")

    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # Calculate timestamp
        secs = frame_count / fps
        timestamp = time.strftime("%H:%M:%S", time.gmtime(secs))

        # Detect motion
        motion = motion_detector.detect(frame_bgr)

        if motion:
            # Convert to RGB PIL Image
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            print(f"\n[MOTION @ {timestamp}] Running investigation...")

            # Tell the frame provider where we are in the video so it can
            # advance frames in real-time during the investigation
            provider_base_frame[0] = frame_count
            provider_wall_start[0] = time.time()

            # Run investigation — takes real wall-clock time (2-5s typically)
            inv_start = time.time()
            result = investigator.investigate_realtime(pil_image, timestamp)
            inv_elapsed = time.time() - inv_start

            # --- Real-world skip ---
            # The investigation took inv_elapsed seconds of wall-clock time.
            # In the real world the camera kept recording during that time.
            # Fast-forward the video by that many seconds so the next motion
            # event comes from AFTER the investigation finished.
            skip_frames = int(inv_elapsed * fps)
            new_frame_pos = frame_count + skip_frames
            if new_frame_pos < total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame_pos)
                frame_count = new_frame_pos
                skip_ts = time.strftime("%H:%M:%S", time.gmtime(new_frame_pos / fps))
                print(f"   >> Video advanced {inv_elapsed:.1f}s "
                      f"(skipped {skip_frames} frames, now at {skip_ts})")
            else:
                # Investigation took longer than remaining video
                break

            # Reset motion detector after skip (new scene context)
            motion_detector.reset()

            # Match to ground truth
            matched_ann = match_investigation_to_annotation(
                timestamp, annotations
            )

            if matched_ann:
                gt_label = matched_ann["label"]
                confusion.add(
                    prediction_status=result["status"],
                    ground_truth_label=gt_label,
                    confidence=result.get("confidence", 0),
                    timestamp=timestamp,
                    matched=True,
                )
                result["ground_truth"] = gt_label
                result["annotation_id"] = matched_ann.get("id")
            else:
                confusion.add(
                    prediction_status=result["status"],
                    ground_truth_label="UNKNOWN",
                    confidence=result.get("confidence", 0),
                    timestamp=timestamp,
                    matched=False,
                )
                result["ground_truth"] = "UNMATCHED"
                result["annotation_id"] = None

            investigation_results.append(result)

        frame_count += 1

        # Progress every 5 seconds of video
        if frame_count % max(1, int(fps * 5)) == 0:
            print(f"   ... processed {timestamp} ({frame_count}/{total_frames} frames)")

    cap.release()
    provider_cap.release()
    print(f"\n{'='*60}")
    print(f"Processed {frame_count} frames, "
          f"{len(investigation_results)} investigations triggered")

    return confusion, metrics_logger, investigation_results


def plot_roc_curve(roc_data: list, output_path: str):
    """Plot ROC curve and save to file."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed, skipping ROC curve plot")
        return

    if not roc_data:
        print("Warning: no ROC data to plot")
        return

    fprs = [p[0] for p in roc_data]
    tprs = [p[1] for p in roc_data]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(fprs, tprs, "b-o", linewidth=2, markersize=4, label="System ROC")
    ax.plot([0, 1], [0, 1], "r--", linewidth=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title("ROC Curve - Violence Detection System")
    ax.legend(loc="lower right")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"ROC curve saved to {output_path}")


def build_report(
    confusion: ConfusionMatrix,
    metrics_logger: MetricsLogger,
    investigation_results: list,
    video_path: str,
    annotations_path: str,
    investigate_as: str,
) -> dict:
    """Build the full benchmark report dict."""
    # ROC data
    roc_data = generate_roc_data(confusion.entries)

    # Confidence distribution
    confidences = [r.get("confidence", 0) for r in investigation_results]
    conf_dist = {}
    if confidences:
        for bucket_start in range(0, 100, 10):
            bucket_end = bucket_start + 10
            key = f"{bucket_start}-{bucket_end}"
            conf_dist[key] = sum(
                1 for c in confidences if bucket_start <= c < bucket_end
            )
        conf_dist["100"] = sum(1 for c in confidences if c == 100)

    return {
        "benchmark_info": {
            "timestamp": datetime.now().isoformat(),
            "video_file": video_path,
            "annotations_file": annotations_path,
            "investigate_as": investigate_as,
        },
        "confusion_matrix": confusion.summary(),
        "roc_curve_data": [{"fpr": p[0], "tpr": p[1]} for p in roc_data],
        "latency_summary": metrics_logger.summary(),
        "confidence_distribution": conf_dist,
        "investigations": [
            {
                "timestamp": r.get("timestamp", ""),
                "status": r.get("status", ""),
                "confidence": r.get("confidence", 0),
                "ground_truth": r.get("ground_truth", ""),
                "annotation_id": r.get("annotation_id"),
                "rounds": r.get("rounds", 0),
                "reason": r.get("reason", ""),
            }
            for r in investigation_results
        ],
    }


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("  VIOLENCE DETECTION BENCHMARK")
    print("=" * 60)

    # Validate
    if not Path(args.video).exists():
        print(f"Error: video not found: {args.video}")
        sys.exit(1)
    if not Config.GROQ_API_KEY:
        print("Error: GROQ_API_KEY not set in .env file")
        sys.exit(1)

    # Load annotations
    ann_data = load_annotations(args.annotations)
    annotations = ann_data["annotations"]

    # Run benchmark
    confusion, metrics_logger, results = run_benchmark(
        args.video, annotations, args.investigate_as
    )

    # Print results
    confusion.print_matrix()
    metrics_logger.print_summary()

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        video_stem = Path(args.video).stem
        output_path = f"results/benchmark_{video_stem}.json"

    # Export JSON report
    report = build_report(
        confusion, metrics_logger, results,
        args.video, args.annotations, args.investigate_as,
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report saved to {output_path}")

    # Plot ROC curve
    roc_path = str(Path(output_path).parent / "roc_curve.png")
    roc_data = generate_roc_data(confusion.entries)
    plot_roc_curve(roc_data, roc_path)

    print(f"\nBenchmark complete. {len(results)} investigations evaluated.")


if __name__ == "__main__":
    main()
