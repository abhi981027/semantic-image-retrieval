"""
Phase 2: video -> sampled frames -> keyframes -> SAM + CLIP ingestion.

Pipeline
    1. Read the video with OpenCV and sample frames at `fps` (default: 1 fps for clips up
       to 2 minutes, 0.5 fps for longer ones).
    2. Keyframe selection: compare each sampled frame with the LAST KEPT keyframe and drop
       it if they are too similar (similarity > similarity_threshold).
         - "histogram": HSV colour-histogram correlation (cv2.compareHist). Fast, no model.
         - "clip":      cosine similarity of CLIP image embeddings. Slower, more semantic.
       A frame is always kept if `max_gap_sec` has passed since the last keyframe, so slow
       scene changes (e.g. dashcam) still get temporal coverage.
    3. Save keyframes as PNGs in frames/<video_name>/frame_<frame_number>.png and write
       frames/<video_name>/keyframes.json.
    4. ingest_video() passes the keyframes to ingest.ingest_image_list(), adding
       {source_video, frame_number, timestamp_sec} to every metadata entry.

Usage
    python video_ingest.py --video videos/clip.mp4                 # extract + ingest
    python video_ingest.py --video videos/clip.mp4 --extract-only  # just keyframes
"""
import os
import re
import json
import time
import argparse
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".m4v")
LONG_VIDEO_SEC = 120.0


def default_sample_fps(duration_sec: float) -> float:
    """1 fps for clips up to 2 min, 0.5 fps for longer ones."""
    return 1.0 if duration_sec <= LONG_VIDEO_SEC else 0.5


def safe_video_name(video_path: str) -> str:
    """Filesystem-safe folder name derived from the video filename."""
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_") or "video"


def probe_video(video_path: str) -> Dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration = frame_count / native_fps if native_fps > 0 and frame_count > 0 else 0.0
    return {"native_fps": native_fps, "frame_count": frame_count,
            "width": width, "height": height, "duration_sec": duration}


# ---------------------------------------------------------------------------
# Frame similarity measures
# ---------------------------------------------------------------------------

def hsv_histogram(frame_bgr: np.ndarray) -> np.ndarray:
    """Normalised 2-D Hue/Saturation histogram (50 x 60 bins)."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist


def histogram_similarity(h1: np.ndarray, h2: np.ndarray) -> float:
    """Correlation in [-1, 1]; 1 = identical colour distribution."""
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))


class ClipFrameEncoder:
    """CLIP image embeddings for keyframe selection. Reuses an already-loaded model if given."""

    def __init__(self, clip_model=None, clip_processor=None, device=None):
        import torch  # noqa: F401  (imported lazily so histogram mode needs no torch)
        from transformers import CLIPModel, CLIPProcessor
        from device_utils import get_device
        from ingest import CLIP_MODEL_NAME
        self.device = device or get_device()
        self.clip_model = clip_model or CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(self.device).eval()
        self.clip_processor = clip_processor or CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        from PIL import Image
        from ingest import clip_image_embeddings
        rgb = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        return clip_image_embeddings([rgb], self.clip_model, self.clip_processor, self.device)[0]


def _resize_max_side(frame: np.ndarray, max_side: Optional[int]) -> np.ndarray:
    if not max_side:
        return frame
    h, w = frame.shape[:2]
    scale = max_side / float(max(h, w))
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Keyframe extraction
# ---------------------------------------------------------------------------

def extract_keyframes(
    video_path: str,
    output_dir: str,
    method: str = "histogram",
    fps: Optional[float] = None,
    similarity_threshold: float = 0.95,
    max_gap_sec: Optional[float] = 10.0,
    max_side: Optional[int] = 1280,
    clip_encoder: Optional[ClipFrameEncoder] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Dict:
    """
    Sample frames from `video_path` and keep only visually distinct keyframes.

    Args:
        output_dir: folder for this video's keyframes (e.g. frames/<video_name>/). Old
            frame_*.png files in it are removed first so re-runs don't mix results.
        method: "histogram" or "clip".
        fps: sampling rate in frames per second. None -> 1.0 (<= 2 min) or 0.5 (longer).
        similarity_threshold: drop a sampled frame if its similarity to the last kept
            keyframe is ABOVE this value. Higher -> more keyframes.
        max_gap_sec: always keep a frame after this many seconds without a keyframe
            (None disables).
        max_side: downscale saved keyframes so the longer side is at most this many pixels
            (None keeps native resolution). SAM resizes to 1024 internally anyway.

    Returns:
        {"video": {...probe info...}, "keyframes": [ {source_video, frame_number,
         timestamp_sec, keyframe_path, similarity_to_prev}, ...], "stats": {...}}
    """
    if method not in ("histogram", "clip"):
        raise ValueError("method must be 'histogram' or 'clip'")

    info = probe_video(video_path)
    native_fps = info["native_fps"]
    sample_fps = fps if fps and fps > 0 else default_sample_fps(info["duration_sec"])

    os.makedirs(output_dir, exist_ok=True)
    for f in os.listdir(output_dir):
        if f.startswith("frame_") and f.endswith(".png"):
            os.remove(os.path.join(output_dir, f))

    if method == "clip" and clip_encoder is None:
        clip_encoder = ClipFrameEncoder()
    describe = hsv_histogram if method == "histogram" else clip_encoder

    def similarity(a, b) -> float:
        if method == "histogram":
            return histogram_similarity(a, b)
        return float(np.dot(a, b))  # CLIP embeddings are L2-normalised -> cosine

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    expected_samples = int(info["duration_sec"] * sample_fps) + 1 if info["duration_sec"] else 0
    keyframes: List[Dict] = []
    last_desc = None
    last_kf_time = None
    frame_idx = -1
    sampled = 0
    next_sample_t = 0.0
    t_decode = t_select = t_save = 0.0
    t_start = time.perf_counter()

    while True:
        t0 = time.perf_counter()
        ok = cap.grab()  # grab() advances without the colour conversion of read()
        if not ok:
            break
        frame_idx += 1
        if native_fps > 0:
            t_sec = frame_idx / native_fps
        else:  # some containers don't report fps; fall back to the decoder clock
            t_sec = (cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
        if t_sec + 1e-6 < next_sample_t:
            t_decode += time.perf_counter() - t0
            continue
        ok, frame = cap.retrieve()
        t_decode += time.perf_counter() - t0
        if not ok or frame is None:
            continue
        next_sample_t += 1.0 / sample_fps
        # If the video jumped past several sample points, re-anchor on the current time.
        if next_sample_t <= t_sec:
            next_sample_t = t_sec + 1.0 / sample_fps
        sampled += 1

        t1 = time.perf_counter()
        frame = _resize_max_side(frame, max_side)
        desc = describe(frame)
        if last_desc is None:
            keep, sim = True, None
        else:
            sim = similarity(desc, last_desc)
            gap_exceeded = max_gap_sec is not None and (t_sec - last_kf_time) >= max_gap_sec
            keep = sim <= similarity_threshold or gap_exceeded
        t_select += time.perf_counter() - t1

        if keep:
            t2 = time.perf_counter()
            kf_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.png")
            cv2.imwrite(kf_path, frame)
            t_save += time.perf_counter() - t2
            keyframes.append({
                "source_video": video_path,
                "frame_number": frame_idx,
                "timestamp_sec": round(t_sec, 3),
                "keyframe_path": kf_path,
                "similarity_to_prev": None if sim is None else round(sim, 4),
            })
            last_desc = desc
            last_kf_time = t_sec

        if progress_cb:
            progress_cb(sampled, max(expected_samples, sampled))

    cap.release()
    total = time.perf_counter() - t_start

    stats = {
        "method": method,
        "sample_fps": sample_fps,
        "similarity_threshold": similarity_threshold,
        "max_gap_sec": max_gap_sec,
        "max_side": max_side,
        "frames_decoded": frame_idx + 1,
        "frames_sampled": sampled,
        "keyframes": len(keyframes),
        # Fraction of sampled frames removed by keyframe selection
        "keyframe_reduction_ratio": (1.0 - len(keyframes) / sampled) if sampled else 0.0,
        "extraction_sec": total,
        "decode_sec": t_decode,
        "selection_sec": t_select,
        "save_sec": t_save,
        "sampled_frames_per_sec": sampled / total if total > 0 else 0.0,
        "decoded_frames_per_sec": (frame_idx + 1) / total if total > 0 else 0.0,
    }
    result = {"video": {"path": video_path, **info}, "keyframes": keyframes, "stats": stats}
    with open(os.path.join(output_dir, "keyframes.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


# ---------------------------------------------------------------------------
# Full video ingestion
# ---------------------------------------------------------------------------

def ingest_video(
    video_path: str,
    index_path: str = "faiss_index.faiss",
    metadata_path: str = "metadata.json",
    frames_root: str = "frames",
    method: str = "histogram",
    fps: Optional[float] = None,
    similarity_threshold: float = 0.95,
    max_gap_sec: Optional[float] = 10.0,
    max_keyframes: Optional[int] = None,
    status_cb: Optional[Callable[[str, int, int], None]] = None,
) -> Dict:
    """
    Extract keyframes from a video and add them to the FAISS index with video metadata.

    status_cb(stage, done, total) is called with stage in {"extracting", "ingesting"}.
    max_keyframes: optional cap (evenly subsampled) to bound ingestion time on CPU.
    """
    from ingest import IngestModels, ingest_image_list, SAM_PARAMS_FAST

    out_dir = os.path.join(frames_root, safe_video_name(video_path))
    models = IngestModels(sam_params=SAM_PARAMS_FAST)
    try:
        encoder = None
        if method == "clip":
            encoder = ClipFrameEncoder(models.clip_model, models.clip_processor, models.device)

        extraction = extract_keyframes(
            video_path, out_dir, method=method, fps=fps,
            similarity_threshold=similarity_threshold, max_gap_sec=max_gap_sec,
            clip_encoder=encoder,
            progress_cb=(lambda d, t: status_cb("extracting", d, t)) if status_cb else None,
        )
        kfs = extraction["keyframes"]
        if max_keyframes and len(kfs) > max_keyframes:
            pick = np.linspace(0, len(kfs) - 1, max_keyframes).round().astype(int)
            kfs = [kfs[i] for i in sorted(set(pick.tolist()))]
        extraction["stats"]["keyframes_ingested"] = len(kfs)

        paths = [k["keyframe_path"] for k in kfs]
        extra = [{"source_video": k["source_video"],
                  "frame_number": k["frame_number"],
                  "timestamp_sec": k["timestamp_sec"]} for k in kfs]
        ingest_stats = ingest_image_list(
            paths, index_path, metadata_path, extra_metadata=extra, append=True, models=models,
            progress_cb=(lambda d, t: status_cb("ingesting", d, t)) if status_cb else None,
        )
    finally:
        models.unload()

    return {"video": extraction["video"], "extraction": extraction["stats"], "ingestion": ingest_stats,
            "frames_dir": out_dir}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Extract keyframes from a video and (optionally) index them.")
    p.add_argument("--video", required=True, help="Path to the video file.")
    p.add_argument("--frames-root", default="frames")
    p.add_argument("--index", default="faiss_index.faiss")
    p.add_argument("--meta", default="metadata.json")
    p.add_argument("--method", choices=["histogram", "clip"], default="histogram")
    p.add_argument("--fps", type=float, default=None, help="Sampling fps (default: 1.0 if <=2 min else 0.5)")
    p.add_argument("--threshold", type=float, default=0.95, help="Similarity above which a frame is skipped")
    p.add_argument("--max-gap", type=float, default=10.0, help="Force a keyframe after this many seconds (0 = off)")
    p.add_argument("--max-keyframes", type=int, default=None)
    p.add_argument("--extract-only", action="store_true", help="Only extract keyframes, don't run SAM/CLIP ingestion")
    a = p.parse_args()
    max_gap = a.max_gap if a.max_gap and a.max_gap > 0 else None

    if a.extract_only:
        res = extract_keyframes(a.video, os.path.join(a.frames_root, safe_video_name(a.video)),
                                method=a.method, fps=a.fps, similarity_threshold=a.threshold,
                                max_gap_sec=max_gap)
        print(json.dumps(res["stats"], indent=2))
    else:
        res = ingest_video(a.video, a.index, a.meta, a.frames_root, a.method, a.fps, a.threshold,
                           max_gap, a.max_keyframes)
        print(json.dumps({"extraction": res["extraction"],
                          "ingestion": {k: v for k, v in res["ingestion"].items() if k != "per_image_sec"}},
                         indent=2))
