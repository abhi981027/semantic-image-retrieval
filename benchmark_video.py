"""
Phase 2 benchmark: keyframe extraction, SAM+CLIP ingestion and search latency.

Everything is written to a separate benchmark index (benchmarks/<run>/), so the app's
faiss_index.faiss / metadata.json are never touched.

Measures per video
    - frames decoded / sampled, extraction throughput (sampled frames per second)
    - keyframes kept and keyframe reduction ratio (1 - keyframes / sampled frames)
    - SAM+CLIP ingestion time per keyframe (mean / median / max) and embeddings per keyframe
Measures once per run
    - search latency: CLIP query encoding and FAISS search timed separately
      (median and p95 over --repeats runs per query, after one warm-up)
    - peak process memory (RSS)
Optional Phase 1 comparison
    - --images DIR ingests a folder of still images with the same code path and reports
      seconds per image, so Phase 1 and Phase 2 numbers come from the same machine and run.

Examples
    # extraction only (fast, no models): compare histogram vs CLIP-free settings
    python benchmark_video.py --videos videos/*.mp4 --extract-only

    # full run on 3 videos, both keyframe methods, with a Phase 1 image baseline
    python benchmark_video.py --videos videos/dashcam.mp4 videos/lecture.mp4 videos/cctv.mp4 \
        --methods histogram clip --images images/ --queries "a red car" "a person walking"

Outputs: benchmarks/<run>/results.json and summary.md
"""
import os
import sys
import json
import time
import glob
import argparse
import platform
import resource
import statistics
from datetime import datetime

import numpy as np

from video_ingest import extract_keyframes, safe_video_name

DEFAULT_QUERIES = [
    "a person walking", "a car on the road", "a person speaking in front of a whiteboard",
    "a laptop on a desk", "a tree", "a door", "text on a screen", "a group of people",
]


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports kilobytes
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024


def pct(xs, q):
    return float(np.percentile(xs, q)) if xs else None


def env_info(device=None) -> dict:
    info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import cv2
        info["opencv"] = cv2.__version__
    except Exception:
        pass
    try:
        import torch
        import faiss
        import transformers
        info.update(torch=torch.__version__, faiss=faiss.__version__, transformers=transformers.__version__)
    except Exception:
        pass
    if device:
        info["device"] = device
    return info


def time_search(queries, index_path, meta_path, repeats, device):
    import faiss
    from transformers import CLIPModel, CLIPProcessor
    from search import encode_text
    from ingest import CLIP_MODEL_NAME

    index = faiss.read_index(index_path)
    t0 = time.perf_counter()
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    load_sec = time.perf_counter() - t0

    encode_ms, faiss_ms = [], []
    encode_text("warm up", model, proc, device)
    for q in queries:
        for _ in range(repeats):
            t1 = time.perf_counter()
            emb = encode_text(q, model, proc, device)
            t2 = time.perf_counter()
            index.search(emb, min(100, index.ntotal))
            t3 = time.perf_counter()
            encode_ms.append((t2 - t1) * 1000)
            faiss_ms.append((t3 - t2) * 1000)
    total = [a + b for a, b in zip(encode_ms, faiss_ms)]
    return {
        "index_vectors": index.ntotal,
        "clip_load_sec": load_sec,
        "queries": len(queries), "repeats": repeats,
        "encode_ms_median": statistics.median(encode_ms), "encode_ms_p95": pct(encode_ms, 95),
        "faiss_ms_median": statistics.median(faiss_ms), "faiss_ms_p95": pct(faiss_ms, 95),
        "total_ms_median": statistics.median(total), "total_ms_p95": pct(total, 95),
    }


def summarise_ingest(stats: dict) -> dict:
    per = stats["per_image_sec"]
    n = max(stats["images"] - stats["failed"], 1)
    return {
        "items": stats["images"], "failed": stats["failed"],
        "embeddings_added": stats["embeddings_added"],
        "embeddings_per_item": stats["embeddings_added"] / n,
        "total_sec": stats["total_sec"],
        "sec_per_item_mean": stats["mean_sec_per_image"],
        "sec_per_item_median": statistics.median(per) if per else None,
        "sec_per_item_max": max(per) if per else None,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos", nargs="+", required=True, help="Video files (globs allowed)")
    p.add_argument("--methods", nargs="+", default=["histogram"], choices=["histogram", "clip"])
    p.add_argument("--fps", type=float, default=None, help="Sampling fps (default: auto)")
    p.add_argument("--threshold", type=float, default=0.95)
    p.add_argument("--max-gap", type=float, default=10.0, help="0 disables")
    p.add_argument("--max-keyframes", type=int, default=None,
                   help="Cap keyframes ingested per video (evenly subsampled) to bound CPU time")
    p.add_argument("--extract-only", action="store_true", help="Skip SAM/CLIP ingestion and search")
    p.add_argument("--images", default=None, help="Phase 1 baseline: folder of still images to ingest")
    p.add_argument("--queries", nargs="+", default=DEFAULT_QUERIES)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--out", default=None, help="Output folder (default benchmarks/<timestamp>)")
    a = p.parse_args()

    videos = sorted({f for pat in a.videos for f in glob.glob(pat)})
    if not videos:
        sys.exit("No videos found.")
    run_dir = a.out or os.path.join("benchmarks", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    max_gap = a.max_gap if a.max_gap and a.max_gap > 0 else None

    models = None
    device = None
    if not a.extract_only:
        from ingest import IngestModels, SAM_PARAMS_FAST
        t0 = time.perf_counter()
        models = IngestModels(sam_params=SAM_PARAMS_FAST)
        model_load_sec = time.perf_counter() - t0
        device = models.device

    report = {"env": env_info(device),
              "config": {"methods": a.methods, "fps": a.fps or "auto", "threshold": a.threshold,
                         "max_gap_sec": max_gap, "max_keyframes": a.max_keyframes},
              "videos": [], "phase1_images": None, "search": {}}
    if models:
        report["config"]["model_load_sec"] = model_load_sec

    for method in a.methods:
        index_path = os.path.join(run_dir, f"index_{method}.faiss")
        meta_path = os.path.join(run_dir, f"metadata_{method}.json")
        encoder = None
        if method == "clip":
            from video_ingest import ClipFrameEncoder
            encoder = ClipFrameEncoder(models.clip_model, models.clip_processor, models.device) if models \
                else ClipFrameEncoder()

        for v in videos:
            print(f"\n=== {os.path.basename(v)} [{method}] ===")
            out_dir = os.path.join(run_dir, "frames", method, safe_video_name(v))
            ex = extract_keyframes(v, out_dir, method=method, fps=a.fps, similarity_threshold=a.threshold,
                                   max_gap_sec=max_gap, clip_encoder=encoder)
            row = {"video": v, "method": method,
                   "duration_sec": ex["video"]["duration_sec"], "native_fps": ex["video"]["native_fps"],
                   "resolution": f'{ex["video"]["width"]}x{ex["video"]["height"]}',
                   "extraction": ex["stats"]}
            print(f'sampled {ex["stats"]["frames_sampled"]}, keyframes {ex["stats"]["keyframes"]}, '
                  f'reduction {ex["stats"]["keyframe_reduction_ratio"]:.1%}, '
                  f'{ex["stats"]["extraction_sec"]:.2f}s')

            if models:
                from ingest import ingest_image_list
                kfs = ex["keyframes"]
                if a.max_keyframes and len(kfs) > a.max_keyframes:
                    pick = np.linspace(0, len(kfs) - 1, a.max_keyframes).round().astype(int)
                    kfs = [kfs[i] for i in sorted(set(pick.tolist()))]
                st = ingest_image_list(
                    [k["keyframe_path"] for k in kfs], index_path, meta_path,
                    extra_metadata=[{"source_video": k["source_video"], "frame_number": k["frame_number"],
                                     "timestamp_sec": k["timestamp_sec"]} for k in kfs],
                    append=True, models=models)
                row["ingestion"] = summarise_ingest(st)
                dur = row["duration_sec"] or 0
                # End-to-end cost of indexing one second of video on this machine
                row["index_sec_per_video_sec"] = ((ex["stats"]["extraction_sec"] + st["total_sec"]) / dur) if dur else None
            report["videos"].append(row)

        if models and os.path.exists(index_path):
            report["search"][method] = time_search(a.queries, index_path, meta_path, a.repeats, device)

    if models and a.images:
        from ingest import ingest_image_list
        imgs = sorted(os.path.join(a.images, f) for f in os.listdir(a.images)
                      if f.lower().endswith((".png", ".jpg", ".jpeg")))
        print(f"\n=== Phase 1 baseline: {len(imgs)} images ===")
        st = ingest_image_list(imgs, os.path.join(run_dir, "index_images.faiss"),
                               os.path.join(run_dir, "metadata_images.json"), append=False, models=models)
        report["phase1_images"] = summarise_ingest(st)
        report["search"]["phase1_images"] = time_search(
            a.queries, os.path.join(run_dir, "index_images.faiss"),
            os.path.join(run_dir, "metadata_images.json"), a.repeats, device)

    if models:
        models.unload()
    report["peak_rss_mb"] = peak_rss_mb()

    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    with open(os.path.join(run_dir, "summary.md"), "w") as f:
        f.write(markdown_summary(report))
    print(f"\nSaved {run_dir}/results.json and {run_dir}/summary.md")
    print(markdown_summary(report))


def _f(x, nd=2):
    return "–" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def markdown_summary(r: dict) -> str:
    e = r["env"]
    lines = ["# EdgeSeek Phase 2 benchmark", "",
             f'- Date: {e["timestamp"]}',
             f'- Machine: {e["platform"]} ({e["machine"]}, {e["cpu_count"]} CPUs)',
             f'- Device: {e.get("device", "n/a (extraction only)")}',
             f'- Versions: python {e["python"]}, opencv {e.get("opencv", "?")}, torch {e.get("torch", "–")}, '
             f'faiss {e.get("faiss", "–")}, transformers {e.get("transformers", "–")}',
             f'- Config: {json.dumps(r["config"])}',
             f'- Peak RSS: {r["peak_rss_mb"]:.0f} MB', "",
             "## Keyframe extraction", "",
             "| Video | Method | Duration (s) | Sample fps | Sampled | Keyframes | Reduction | Extract (s) | Sampled frames/s |",
             "|---|---|---|---|---|---|---|---|---|"]
    for v in r["videos"]:
        s = v["extraction"]
        lines.append(f'| {os.path.basename(v["video"])} | {v["method"]} | {_f(v["duration_sec"], 1)} | '
                     f'{_f(s["sample_fps"], 1)} | {s["frames_sampled"]} | {s["keyframes"]} | '
                     f'{s["keyframe_reduction_ratio"] * 100:.1f}% | {_f(s["extraction_sec"])} | '
                     f'{_f(s["sampled_frames_per_sec"], 1)} |')
    if any("ingestion" in v for v in r["videos"]):
        lines += ["", "## SAM + CLIP ingestion", "",
                  "| Video | Method | Keyframes ingested | Embeddings/keyframe | s/keyframe (mean) | s/keyframe (median) | Total (s) | Index s per video s |",
                  "|---|---|---|---|---|---|---|---|"]
        for v in r["videos"]:
            g = v.get("ingestion")
            if not g:
                continue
            lines.append(f'| {os.path.basename(v["video"])} | {v["method"]} | {g["items"]} | '
                         f'{_f(g["embeddings_per_item"], 1)} | {_f(g["sec_per_item_mean"])} | '
                         f'{_f(g["sec_per_item_median"])} | {_f(g["total_sec"], 1)} | '
                         f'{_f(v.get("index_sec_per_video_sec"))} |')
    if r.get("phase1_images"):
        g = r["phase1_images"]
        lines += ["", "## Phase 1 baseline (still images, same run)", "",
                  f'- Images: {g["items"]}, embeddings/image: {_f(g["embeddings_per_item"], 1)}',
                  f'- s/image: mean {_f(g["sec_per_item_mean"])}, median {_f(g["sec_per_item_median"])}, '
                  f'max {_f(g["sec_per_item_max"])}']
    if r["search"]:
        lines += ["", "## Search latency (text query, excludes per-request model load)", "",
                  "| Index | Vectors | CLIP encode median (ms) | FAISS median (ms) | Total median (ms) | Total p95 (ms) |",
                  "|---|---|---|---|---|---|"]
        for name, s in r["search"].items():
            lines.append(f'| {name} | {s["index_vectors"]} | {_f(s["encode_ms_median"])} | '
                         f'{_f(s["faiss_ms_median"], 3)} | {_f(s["total_ms_median"])} | {_f(s["total_ms_p95"])} |')
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
