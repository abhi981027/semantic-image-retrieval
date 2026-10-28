import os
import shutil
import json
import threading
import time
import uuid
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from ingest import ingest_single_image
from search import search_by_text, search_by_image, format_timestamp, sort_timeline
from video_ingest import ingest_video, VIDEO_EXTENSIONS
from device_utils import get_device, free_device_memory
import faiss
from transformers import CLIPProcessor, CLIPModel

app = FastAPI()

# Configuration
IMAGES_DIR = "images"
VIDEOS_DIR = "videos"
FRAMES_DIR = "frames"
INDEX_PATH = "faiss_index.faiss"
METADATA_PATH = "metadata.json"
STATIC_DIR = "static"
MIN_SIMILARITY = 0.2  # filter weak matches when using cosine similarity (threshold)
FETCH_K = 100         # fetch more candidates, then re-rank/diversify
SCORE_TIE_EPS = 0.02  # prefer global over object if scores are very close

# Video defaults (can be overridden per upload from the UI)
VIDEO_KEYFRAME_METHOD = "histogram"
VIDEO_SIMILARITY_THRESHOLD = 0.95
VIDEO_MAX_GAP_SEC = 10.0

# Ensure directories exist
for d in (IMAGES_DIR, VIDEOS_DIR, FRAMES_DIR, STATIC_DIR, os.path.join(STATIC_DIR, "uploads")):
    os.makedirs(d, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
# Keyframes are served directly from frames/ (no copying, no filename collisions between videos)
app.mount("/frames", StaticFiles(directory=FRAMES_DIR), name="frames")

# We load models on-demand for search to save memory.
# The index and metadata are loaded once at startup and reloaded after ingestion.
index = None
metadata = {}
clip_model = None
clip_processor = None
device = "cpu"

# Only one ingestion (image or video) may write the index at a time.
ingest_lock = threading.Lock()
# job_id -> status dict for background video ingestion
video_jobs: dict = {}


def load_search_models():
    """Loads CLIP model and processor for search."""
    global clip_model, clip_processor, device
    device = get_device()
    clip_model_name = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    clip_model.eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    print(f"CLIP model loaded for search on {device}.")

def unload_search_models():
    """Unloads models to free up memory."""
    global clip_model, clip_processor
    del clip_model
    del clip_processor
    clip_model = None
    clip_processor = None
    free_device_memory(device)
    print("CLIP model unloaded.")

def load_index_and_metadata():
    """Loads or reloads the FAISS index and metadata from disk."""
    global index, metadata
    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
        print("FAISS index loaded.")
    else:
        index = None
        print("FAISS index not found.")

    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        print("Metadata loaded.")
    else:
        metadata = {}
        print("Metadata not found.")

# Initial load at startup
load_index_and_metadata()


def indexed_videos() -> dict:
    """source_video -> {keyframes, duration_sec} for every video in the index."""
    videos: dict = {}
    for meta in metadata.values():
        src = meta.get("source_video")
        if not src or meta.get("type") != "global":
            continue
        v = videos.setdefault(src, {"keyframes": 0, "last_timestamp_sec": 0.0})
        v["keyframes"] += 1
        v["last_timestamp_sec"] = max(v["last_timestamp_sec"], meta.get("timestamp_sec") or 0.0)
    return videos


@app.get("/")
def read_root():
    return FileResponse(os.path.join(STATIC_DIR, 'index.html'))

# Plain `def` (not async) so FastAPI runs this in a worker thread: ingestion is blocking and
# must not freeze the server while a video job is polling for progress.
@app.post("/upload/")
def upload_images(files: list[UploadFile] = File(...)):
    ingested_files_count = 0
    skipped_files_count = 0

    for file in files:
        file_path = os.path.join(IMAGES_DIR, os.path.basename(file.filename))

        # Check if the image is already indexed
        if any(meta.get('image_path') == file_path for meta in metadata.values()):
            skipped_files_count += 1
            continue

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run ingestion for the new image
        print(f"Starting ingestion for {file.filename}...")
        with ingest_lock:
            ingest_single_image(file_path, INDEX_PATH, METADATA_PATH)
        ingested_files_count += 1

    # Reload data only if new files were added
    if ingested_files_count > 0:
        load_index_and_metadata()

    message = f"Processing complete. Ingested: {ingested_files_count} new image(s). Skipped: {skipped_files_count} existing image(s)."
    return JSONResponse(content={"message": message})


def _run_video_job(job_id: str, video_path: str, method: str, fps, threshold: float,
                   max_gap, max_keyframes):
    job = video_jobs[job_id]

    def status_cb(stage, done, total):
        job.update(stage=stage, done=done, total=total)

    try:
        job["stage"] = "waiting"  # another ingestion may hold the lock
        with ingest_lock:
            job["stage"] = "loading_models"
            result = ingest_video(
                video_path, INDEX_PATH, METADATA_PATH, FRAMES_DIR,
                method=method, fps=fps, similarity_threshold=threshold,
                max_gap_sec=max_gap, max_keyframes=max_keyframes, status_cb=status_cb,
            )
        load_index_and_metadata()
        ex, ing = result["extraction"], result["ingestion"]
        job.update(
            stage="done",
            finished_at=time.time(),
            frames_sampled=ex["frames_sampled"],
            keyframes=ex["keyframes"],
            keyframes_ingested=ex.get("keyframes_ingested", ex["keyframes"]),
            keyframe_reduction_ratio=ex["keyframe_reduction_ratio"],
            sample_fps=ex["sample_fps"],
            duration_sec=result["video"]["duration_sec"],
            extraction_sec=ex["extraction_sec"],
            ingestion_sec=ing["total_sec"],
            mean_sec_per_keyframe=ing["mean_sec_per_image"],
            embeddings_added=ing["embeddings_added"],
            message=(f"Indexed {ex.get('keyframes_ingested', ex['keyframes'])} keyframes "
                     f"(from {ex['frames_sampled']} sampled frames, "
                     f"{ex['keyframe_reduction_ratio'] * 100:.0f}% reduction)."),
        )
    except Exception as e:
        print(f"Video job {job_id} failed: {e}")
        job.update(stage="error", message=str(e), finished_at=time.time())


@app.post("/upload-video/")
def upload_video(
    file: UploadFile = File(...),
    method: str = Form(VIDEO_KEYFRAME_METHOD),
    fps: float = Form(0.0),                       # 0 -> automatic (1 fps <= 2 min, else 0.5 fps)
    similarity_threshold: float = Form(VIDEO_SIMILARITY_THRESHOLD),
    max_gap_sec: float = Form(VIDEO_MAX_GAP_SEC),  # 0 -> disabled
    max_keyframes: int = Form(0),                 # 0 -> no cap
):
    """
    Saves the video, then extracts keyframes and runs SAM + CLIP in a background thread.
    Returns a job_id immediately; poll GET /video-status/{job_id} for progress.
    """
    name = os.path.basename(file.filename or "")
    if not name.lower().endswith(VIDEO_EXTENSIONS):
        raise HTTPException(status_code=400, detail=f"Unsupported video type. Use one of {', '.join(VIDEO_EXTENSIONS)}.")
    if method not in ("histogram", "clip"):
        raise HTTPException(status_code=400, detail="method must be 'histogram' or 'clip'.")
    if not 0.0 < similarity_threshold <= 1.0:
        raise HTTPException(status_code=400, detail="similarity_threshold must be in (0, 1].")

    video_path = os.path.join(VIDEOS_DIR, name)
    if video_path in indexed_videos():
        return JSONResponse(content={"job_id": None, "stage": "skipped",
                                     "message": f"{name} is already indexed. Rename the file to index it again."})
    if any(j["video"] == video_path and j["stage"] not in ("done", "error") for j in video_jobs.values()):
        raise HTTPException(status_code=409, detail=f"{name} is already being processed.")

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    job_id = uuid.uuid4().hex[:12]
    video_jobs[job_id] = {"job_id": job_id, "video": video_path, "video_name": name, "stage": "queued",
                          "done": 0, "total": 0, "started_at": time.time(), "method": method}
    threading.Thread(
        target=_run_video_job,
        args=(job_id, video_path, method, fps if fps > 0 else None, similarity_threshold,
              max_gap_sec if max_gap_sec > 0 else None, max_keyframes if max_keyframes > 0 else None),
        daemon=True,
    ).start()
    return JSONResponse(content=video_jobs[job_id])


@app.get("/video-status/{job_id}")
def video_status(job_id: str):
    job = video_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id.")
    return JSONResponse(content={**job, "elapsed_sec": (job.get("finished_at") or time.time()) - job["started_at"]})


@app.get("/videos/")
def list_videos():
    vids = indexed_videos()
    return JSONResponse(content={"videos": [
        {"source_video": k, "name": os.path.basename(k), **v} for k, v in sorted(vids.items())
    ]})


def _result_image_url(image_path: str) -> str:
    """Keyframes are served from /frames; Phase 1 images are copied into static/uploads as before."""
    rel = os.path.relpath(image_path, FRAMES_DIR)
    if not rel.startswith(".."):
        return "/frames/" + rel.replace("\\", "/")
    filename = os.path.basename(image_path)
    static_path = os.path.join(STATIC_DIR, "uploads", filename)
    if not os.path.exists(static_path):
        shutil.copy(image_path, static_path)
    return "/static/uploads/" + filename


@app.post("/search/")
async def search(query_text: str = Form(None), query_image: UploadFile = File(None),
                 source_filter: str = Form(None), sort_by: str = Form("score")):
    if index is None:
        raise HTTPException(status_code=404, detail="Index not found. Please upload and ingest images first.")
    if sort_by not in ("score", "time"):
        sort_by = "score"

    results = []
    t0 = time.perf_counter()
    try:
        load_search_models()
        t_loaded = time.perf_counter()
        if query_text:
            results = search_by_text(query_text, index, metadata, clip_model, clip_processor, device,
                                     top_k=FETCH_K, min_similarity=MIN_SIMILARITY, source_filter=source_filter)
        elif query_image:
            # Save the uploaded query image temporarily to pass its path
            temp_image_path = os.path.join(STATIC_DIR, "uploads", "temp_" + os.path.basename(query_image.filename))
            with open(temp_image_path, "wb") as buffer:
                shutil.copyfileobj(query_image.file, buffer)
            try:
                results = search_by_image(temp_image_path, index, metadata, clip_model, clip_processor, device,
                                          top_k=FETCH_K, min_similarity=MIN_SIMILARITY, source_filter=source_filter)
            finally:
                # Clean up the temporary file
                os.remove(temp_image_path)
        else:
            raise HTTPException(status_code=400, detail="Please provide a text query or an image.")
        t_searched = time.perf_counter()
    finally:
        unload_search_models()


    # Diversify: keep the best (prefer global on ties) per image. For videos, each keyframe
    # is its own image, so this keeps one box per keyframe.
    best_by_image = {}
    for r in results:
        meta = r.get('metadata') or {}
        img = meta.get('image_path')
        if not img:
            continue
        prev = best_by_image.get(img)
        if prev is None:
            best_by_image[img] = r
            continue
        better = r['score'] > prev['score'] + SCORE_TIE_EPS
        tie_and_global_better = abs(r['score'] - prev['score']) <= SCORE_TIE_EPS and meta.get('type') == 'global' and (prev.get('metadata') or {}).get('type') != 'global'
        if better or tie_and_global_better:
            best_by_image[img] = r

    # Final results sorted by score (all results above MIN_SIMILARITY threshold)
    final_results = sorted(best_by_image.values(), key=lambda r: r['score'], reverse=True)
    if sort_by == "time":
        final_results = sort_timeline(final_results)

    # Prepare results for the frontend
    output_results = []
    for res in final_results:
        meta = res['metadata']
        item = {
            "score": res['score'],
            "type": meta['type'],
            "box": meta['box'],
            "image_url": _result_image_url(meta['image_path']),
        }
        if meta.get("source_video"):
            item.update({
                "source_video": meta["source_video"],
                "video_name": os.path.basename(meta["source_video"]),
                "frame_number": meta.get("frame_number"),
                "timestamp_sec": meta.get("timestamp_sec"),
                "timestamp": format_timestamp(meta.get("timestamp_sec")),
            })
        output_results.append(item)

    return JSONResponse(content={
        "results": output_results,
        # model load is reported separately: CLIP is loaded per request to save memory
        "timing": {"model_load_ms": (t_loaded - t0) * 1000, "search_ms": (t_searched - t_loaded) * 1000},
    })
