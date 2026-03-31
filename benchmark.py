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
        default=None,
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
    parser.add_argument(
        "--review-as",
        type=str,
        choices=["VIOLENCE", "NO_VIOLENCE", "EXCLUDE"],
        default="EXCLUDE",
        help="How to map REVIEW status (default: EXCLUDE)",
    )
    parser.add_argument(
        "--quant-mode",
        type=str,
        choices=["auto", "8bit", "4bit", "none"],
        default=Config.EDGE_QUANT_MODE,
        help="Quantization mode for the edge model (default: config value)",
    )
    parser.add_argument(
        "--profile-edge-only",
        action="store_true",
        help="Skip cloud benchmarking and profile edge latency only",
    )
    parser.add_argument(
        "--profile-quant-modes",
        type=str,
        default="8bit,4bit",
        help="Comma-separated quant modes to profile in edge-only mode",
    )
    parser.add_argument(
        "--profile-runs",
        type=int,
        default=3,
        help="Number of timed runs per quant mode in edge-only mode",
    )
    parser.add_argument(
        "--profile-question",
        type=str,
        default=AIInvestigator.INITIAL_EDGE_QUESTION,
        help="Question used for edge-only profiling",
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


def _stats(values: list[float]) -> dict:
    if not values:
        return {"mean": 0, "min": 0, "max": 0}
    return {
        "mean": round(sum(values) / len(values), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "runs": [round(v, 2) for v in values],
    }


def _load_profile_frame(video_path: str) -> Image.Image:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {video_path}")

    try:
        ret, frame_bgr = cap.read()
        if not ret:
            raise RuntimeError(f"cannot read frame from {video_path}")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)
    finally:
        cap.release()


def _time_edge_call(fn):
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000, result


def profile_edge_modes(
    video_path: str,
    quant_modes: list[str],
    runs: int,
    question: str,
):
    import torch

    image = _load_profile_frame(video_path)
    profiles = []

    for requested_mode in quant_modes:
        mode = requested_mode.strip()
        if not mode:
            continue

        print(f"\n[EDGE PROFILE] Loading quant mode: {mode}")
        edge = EdgeVision(
            model_id=Config.EDGE_MODEL_ID,
            device=Config.DEVICE,
            quant_mode=mode,
        )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        edge.analyze(
            image,
            question,
            max_tokens=Config.EDGE_INITIAL_MAX_TOKENS,
        )
        edge.clear_cache()

        encode_runs = []
        answer_initial_runs = []
        answer_followup_runs = []
        analyze_runs = []
        answer_preview = ""

        for run_idx in range(runs):
            edge.clear_cache()
            encode_ms, encoded = _time_edge_call(lambda: edge.encode(image))
            encode_runs.append(encode_ms)

            answer_initial_ms, initial_answer = _time_edge_call(
                lambda: edge.answer(
                    encoded,
                    question,
                    max_tokens=Config.EDGE_INITIAL_MAX_TOKENS,
                )
            )
            answer_initial_runs.append(answer_initial_ms)

            answer_followup_ms, _ = _time_edge_call(
                lambda: edge.answer(
                    encoded,
                    question,
                    max_tokens=Config.EDGE_FOLLOWUP_MAX_TOKENS,
                )
            )
            answer_followup_runs.append(answer_followup_ms)

            edge.clear_cache()
            analyze_ms, analyze_answer = _time_edge_call(
                lambda: edge.analyze(
                    image,
                    question,
                    max_tokens=Config.EDGE_INITIAL_MAX_TOKENS,
                )
            )
            analyze_runs.append(analyze_ms)

            if run_idx == 0:
                answer_preview = analyze_answer[:160]

        peak_gpu_mb = 0.0
        if torch.cuda.is_available():
            peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        profiles.append(
            {
                "quant_mode_requested": mode,
                "quant_mode_loaded": edge.loaded_quant_mode,
                "device_info": edge.get_device_info(),
                "encode_ms": _stats(encode_runs),
                "answer_initial_ms": _stats(answer_initial_runs),
                "answer_followup_ms": _stats(answer_followup_runs),
                "analyze_ms": _stats(analyze_runs),
                "peak_gpu_mb": round(peak_gpu_mb, 2),
                "answer_preview": answer_preview,
            }
        )

        edge.clear_cache()

    return {
        "profile_info": {
            "timestamp": datetime.now().isoformat(),
            "video_file": video_path,
            "device": Config.DEVICE,
            "profile_question": question,
            "runs": runs,
        },
        "edge_profiles": profiles,
    }


def run_benchmark(
    video_path: str,
    annotations: list,
    investigate_as: str,
    review_as: str,
    quant_mode: str,
):
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
        review_as: How to map REVIEW status

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
    edge_vision = EdgeVision(
        model_id=Config.EDGE_MODEL_ID,
        device=Config.DEVICE,
        quant_mode=quant_mode,
    )
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
        initial_edge_max_tokens=Config.EDGE_INITIAL_MAX_TOKENS,
        followup_edge_max_tokens=Config.EDGE_FOLLOWUP_MAX_TOKENS,
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
    confusion = ConfusionMatrix(
        investigate_as=investigate_as,
        review_as=review_as,
    )
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
    review_as: str,
    quant_mode: str,
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
            "review_as": review_as,
            "quant_mode": quant_mode,
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

    if args.profile_edge_only:
        quant_modes = [
            mode.strip()
            for mode in args.profile_quant_modes.split(",")
            if mode.strip()
        ]
        report = profile_edge_modes(
            args.video,
            quant_modes,
            args.profile_runs,
            args.profile_question,
        )

        if args.output:
            output_path = args.output
        else:
            video_stem = Path(args.video).stem
            output_path = f"results/edge_profile_{video_stem}.json"

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(json.dumps(report, indent=2, default=str))
        print(f"\nEdge profile saved to {output_path}")
        return

    if not args.annotations:
        print("Error: --annotations is required unless --profile-edge-only is set")
        sys.exit(1)
    if not Config.GROQ_API_KEY:
        print("Error: GROQ_API_KEY not set in .env file")
        sys.exit(1)

    # Load annotations
    ann_data = load_annotations(args.annotations)
    annotations = ann_data["annotations"]

    # Run benchmark
    confusion, metrics_logger, results = run_benchmark(
        args.video,
        annotations,
        args.investigate_as,
        args.review_as,
        args.quant_mode,
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
        args.video,
        args.annotations,
        args.investigate_as,
        args.review_as,
        args.quant_mode,
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
