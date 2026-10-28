"""
Device selection helpers shared by ingest.py, search.py, video_ingest.py and app/main.py.

Order of preference: CUDA (NVIDIA GPU) -> MPS (Apple Silicon GPU) -> CPU.

Set the environment variable EDGESEEK_DEVICE=cpu (or cuda / mps) to force a device,
e.g. when benchmarking CPU vs MPS on the M2 MacBook.
"""
import os
import torch


def get_device() -> str:
    forced = os.environ.get("EDGESEEK_DEVICE")
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def free_device_memory(device: str) -> None:
    """Release cached GPU memory after models are deleted."""
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
