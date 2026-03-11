"""Metrics Logger for Violence Detection Pipeline Benchmarking."""

import csv
import json
import time
import statistics
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


class MetricsLogger:
    """
    Instruments the investigation pipeline to track per-investigation
    and system-wide metrics for benchmarking and optimization.
    """

    def __init__(self):
        self._investigations: Dict[int, dict] = {}
        self._system_samples: List[dict] = []
        self._next_id = 1

    def start_investigation(self, video_file: str, trigger_ts: str) -> int:
        """Start tracking a new investigation. Returns investigation ID."""
        inv_id = self._next_id
        self._next_id += 1
        self._investigations[inv_id] = {
            "investigation_id": inv_id,
            "video_file": video_file,
            "trigger_timestamp": trigger_ts,
            "wall_clock_start": time.time(),
            "wall_clock_start_iso": datetime.now().isoformat(),
            "edge_times_ms": [],
            "cloud_times_ms": [],
            "edge_descriptions": [],
            "cloud_decisions": [],
            "estimated_prompt_tokens": [],
            "final_status": None,
            "final_confidence": None,
            "rounds": 0,
            "total_investigation_ms": None,
        }
        return inv_id

    def record_edge_call(self, inv_id: int, duration_s: float, description: str):
        """Record an edge model call's timing and output."""
        if inv_id not in self._investigations:
            return
        inv = self._investigations[inv_id]
        inv["edge_times_ms"].append(round(duration_s * 1000, 2))
        inv["edge_descriptions"].append(description)

    def record_cloud_call(self, inv_id: int, duration_s: float,
                          decision: dict, prompt_len: int):
        """Record a cloud model call's timing, decision, and token estimate."""
        if inv_id not in self._investigations:
            return
        inv = self._investigations[inv_id]
        inv["cloud_times_ms"].append(round(duration_s * 1000, 2))
        inv["cloud_decisions"].append(decision)
        inv["estimated_prompt_tokens"].append(prompt_len // 4)

    def end_investigation(self, inv_id: int, final_status: str,
                          final_confidence: int, rounds: int):
        """Finalize an investigation record."""
        if inv_id not in self._investigations:
            return
        inv = self._investigations[inv_id]
        inv["final_status"] = final_status
        inv["final_confidence"] = final_confidence
        inv["rounds"] = rounds
        inv["total_investigation_ms"] = round(
            (time.time() - inv["wall_clock_start"]) * 1000, 2
        )

    def sample_system_metrics(self):
        """Snapshot current system resource usage."""
        sample = {
            "timestamp": datetime.now().isoformat(),
            "ram_usage_mb": None,
            "gpu_mem_allocated_mb": None,
            "gpu_mem_reserved_mb": None,
        }

        try:
            import psutil
            process = psutil.Process()
            sample["ram_usage_mb"] = round(
                process.memory_info().rss / (1024 * 1024), 2
            )
        except ImportError:
            pass

        try:
            import torch
            if torch.cuda.is_available():
                sample["gpu_mem_allocated_mb"] = round(
                    torch.cuda.memory_allocated() / (1024 * 1024), 2
                )
                sample["gpu_mem_reserved_mb"] = round(
                    torch.cuda.memory_reserved() / (1024 * 1024), 2
                )
        except ImportError:
            pass

        self._system_samples.append(sample)

    def get_investigations(self) -> List[dict]:
        """Return all completed investigation records."""
        return [
            inv for inv in self._investigations.values()
            if inv["final_status"] is not None
        ]

    def summary(self) -> dict:
        """Compute aggregate statistics across all investigations."""
        completed = self.get_investigations()
        if not completed:
            return {"total_investigations": 0}

        # Collect all timing lists
        all_edge_ms = []
        all_cloud_ms = []
        all_total_ms = []
        all_tokens = []
        status_counts = {"CLEAR": 0, "INVESTIGATE": 0, "ALERT": 0}
        confidence_vals = []

        for inv in completed:
            all_edge_ms.extend(inv["edge_times_ms"])
            all_cloud_ms.extend(inv["cloud_times_ms"])
            if inv["total_investigation_ms"] is not None:
                all_total_ms.append(inv["total_investigation_ms"])
            all_tokens.extend(inv["estimated_prompt_tokens"])
            status = inv["final_status"]
            if status in status_counts:
                status_counts[status] += 1
            if inv["final_confidence"] is not None:
                confidence_vals.append(inv["final_confidence"])

        def _stats(values: list) -> dict:
            if not values:
                return {"mean": 0, "p50": 0, "p95": 0, "min": 0, "max": 0}
            sorted_v = sorted(values)
            p95_idx = int(len(sorted_v) * 0.95)
            return {
                "mean": round(statistics.mean(sorted_v), 2),
                "p50": round(statistics.median(sorted_v), 2),
                "p95": round(sorted_v[min(p95_idx, len(sorted_v) - 1)], 2),
                "min": round(min(sorted_v), 2),
                "max": round(max(sorted_v), 2),
            }

        # Peak memory from system samples
        peak_ram = 0
        peak_gpu = 0
        for s in self._system_samples:
            if s["ram_usage_mb"] and s["ram_usage_mb"] > peak_ram:
                peak_ram = s["ram_usage_mb"]
            if s["gpu_mem_allocated_mb"] and s["gpu_mem_allocated_mb"] > peak_gpu:
                peak_gpu = s["gpu_mem_allocated_mb"]

        return {
            "total_investigations": len(completed),
            "status_distribution": status_counts,
            "edge_latency_ms": _stats(all_edge_ms),
            "cloud_latency_ms": _stats(all_cloud_ms),
            "total_investigation_ms": _stats(all_total_ms),
            "estimated_tokens_per_call": _stats(all_tokens),
            "total_tokens_consumed": sum(all_tokens),
            "confidence": _stats(confidence_vals),
            "avg_rounds": round(
                statistics.mean([inv["rounds"] for inv in completed]), 2
            ) if completed else 0,
            "peak_ram_mb": peak_ram,
            "peak_gpu_mb": peak_gpu,
        }

    def export_json(self, path: str):
        """Export all investigation records and system samples to JSON."""
        data = {
            "exported_at": datetime.now().isoformat(),
            "summary": self.summary(),
            "investigations": self.get_investigations(),
            "system_samples": self._system_samples,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"[METRICS] Exported JSON to {path}")

    def export_csv(self, path: str):
        """Export investigation records to CSV (flattened)."""
        completed = self.get_investigations()
        if not completed:
            print("[METRICS] No completed investigations to export.")
            return

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "investigation_id", "video_file", "trigger_timestamp",
            "wall_clock_start_iso", "rounds", "final_status",
            "final_confidence", "total_investigation_ms",
            "mean_edge_ms", "mean_cloud_ms", "total_tokens",
        ]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for inv in completed:
                edge_ms = inv["edge_times_ms"]
                cloud_ms = inv["cloud_times_ms"]
                tokens = inv["estimated_prompt_tokens"]
                writer.writerow({
                    "investigation_id": inv["investigation_id"],
                    "video_file": inv["video_file"],
                    "trigger_timestamp": inv["trigger_timestamp"],
                    "wall_clock_start_iso": inv["wall_clock_start_iso"],
                    "rounds": inv["rounds"],
                    "final_status": inv["final_status"],
                    "final_confidence": inv["final_confidence"],
                    "total_investigation_ms": inv["total_investigation_ms"],
                    "mean_edge_ms": round(statistics.mean(edge_ms), 2) if edge_ms else 0,
                    "mean_cloud_ms": round(statistics.mean(cloud_ms), 2) if cloud_ms else 0,
                    "total_tokens": sum(tokens),
                })
        print(f"[METRICS] Exported CSV to {path}")

    def print_summary(self):
        """Print a formatted summary to console."""
        s = self.summary()
        if s["total_investigations"] == 0:
            print("\n[METRICS] No investigations recorded.")
            return

        print(f"\n{'='*60}")
        print(f"  BENCHMARK SUMMARY")
        print(f"{'='*60}")
        print(f"  Total investigations: {s['total_investigations']}")
        print(f"  Status distribution:  {s['status_distribution']}")
        print(f"  Average rounds:       {s['avg_rounds']}")
        print(f"\n  --- Latency (ms) ---")
        print(f"  Edge   : mean={s['edge_latency_ms']['mean']}, "
              f"p50={s['edge_latency_ms']['p50']}, "
              f"p95={s['edge_latency_ms']['p95']}")
        print(f"  Cloud  : mean={s['cloud_latency_ms']['mean']}, "
              f"p50={s['cloud_latency_ms']['p50']}, "
              f"p95={s['cloud_latency_ms']['p95']}")
        print(f"  Total  : mean={s['total_investigation_ms']['mean']}, "
              f"p50={s['total_investigation_ms']['p50']}, "
              f"p95={s['total_investigation_ms']['p95']}")
        print(f"\n  --- Tokens ---")
        print(f"  Per call: mean={s['estimated_tokens_per_call']['mean']}")
        print(f"  Total consumed: {s['total_tokens_consumed']}")
        print(f"\n  --- Memory ---")
        print(f"  Peak RAM: {s['peak_ram_mb']} MB")
        print(f"  Peak GPU: {s['peak_gpu_mb']} MB")
        print(f"\n  --- Confidence ---")
        print(f"  mean={s['confidence']['mean']}, "
              f"p50={s['confidence']['p50']}, "
              f"p95={s['confidence']['p95']}")
        print(f"{'='*60}\n")
