
import os
import faiss
import json
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import argparse
from typing import Dict, List, Optional

from device_utils import get_device

VIDEO_FIELDS = ("source_video", "frame_number", "timestamp_sec")


def encode_text(query_text, clip_model, clip_processor, device) -> np.ndarray:
    inputs = clip_processor(text=query_text, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)
    return text_features.cpu().numpy().astype('float32')


def encode_image(image_path, clip_model, clip_processor, device) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    # Force channels_last to avoid ambiguity on very small images
    inputs = clip_processor(images=image, return_tensors="pt", input_data_format="channels_last").to(device)
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
        image_features = torch.nn.functional.normalize(image_features, dim=-1)
    return image_features.cpu().numpy().astype('float32')


def matches_source(meta: Dict, source_filter: Optional[str]) -> bool:
    """
    source_filter:
        None / ""   -> everything
        "images"    -> only Phase 1 image entries (no source_video)
        "videos"    -> only video keyframe entries
        otherwise   -> entries whose source_video equals the filter, either as the full
                       stored path or just the filename (e.g. "dashcam.mp4").
    """
    if not source_filter:
        return True
    src = meta.get("source_video")
    if source_filter == "images":
        return src is None
    if source_filter == "videos":
        return src is not None
    if src is None:
        return False
    return src == source_filter or os.path.basename(src) == os.path.basename(source_filter)


def sort_timeline(results: List[Dict]) -> List[Dict]:
    """Group by video (in order of each video's best score), then by timestamp; images go last by score."""
    video_best: Dict[str, float] = {}
    for r in results:
        src = r.get("source_video")
        if src is not None:
            video_best[src] = max(video_best.get(src, -2.0), r["score"])
    video_rank = {v: i for i, v in enumerate(sorted(video_best, key=lambda v: -video_best[v]))}
    videos = [r for r in results if r.get("source_video") is not None]
    images = [r for r in results if r.get("source_video") is None]
    videos.sort(key=lambda r: (video_rank[r["source_video"]], r.get("timestamp_sec") or 0.0))
    images.sort(key=lambda r: -r["score"])
    return videos + images


def search_embedding(query_embedding, index, metadata, top_k=50, min_similarity: float | None = None,
                     source_filter: Optional[str] = None, sort_by: str = "score") -> List[Dict]:
    """
    FAISS inner-product search on a normalised query embedding.

    With a source_filter we search the whole index and filter afterwards: IndexFlatIP scans
    every vector anyway, so this only adds the cost of sorting, and guarantees that matches
    inside the requested video are not cut off by top_k matches from other sources.
    """
    if index is None or index.ntotal == 0:
        return []
    k = index.ntotal if source_filter else min(top_k, index.ntotal)
    sims, indices = index.search(query_embedding, k)

    results = []
    for sim, idx in zip(sims[0], indices[0]):
        if idx < 0:
            continue
        result_metadata = metadata.get(str(int(idx)))
        if result_metadata is None:
            continue
        sim = float(sim)
        if min_similarity is not None and sim < min_similarity:
            continue
        if not matches_source(result_metadata, source_filter):
            continue
        r = {
            "id": int(idx),
            "score": sim,  # cosine similarity in [-1,1]
            "metadata": result_metadata,
        }
        # Surface video fields at top level (absent for image-only entries)
        for f in VIDEO_FIELDS:
            if f in result_metadata:
                r[f] = result_metadata[f]
        results.append(r)
        if len(results) >= top_k:
            break

    if sort_by == "time":
        results = sort_timeline(results)
    return results


def search_by_text(query_text, index, metadata, clip_model, clip_processor, device, top_k=50,
                   min_similarity: float | None = None, source_filter: Optional[str] = None,
                   sort_by: str = "score"):
    q = encode_text(query_text, clip_model, clip_processor, device)
    return search_embedding(q, index, metadata, top_k, min_similarity, source_filter, sort_by)


def search_by_image(image_path, index, metadata, clip_model, clip_processor, device, top_k=50,
                    min_similarity: float | None = None, source_filter: Optional[str] = None,
                    sort_by: str = "score"):
    q = encode_image(image_path, clip_model, clip_processor, device)
    return search_embedding(q, index, metadata, top_k, min_similarity, source_filter, sort_by)


def format_timestamp(sec: Optional[float]) -> Optional[str]:
    """42.0 -> '0:42', 3725 -> '1:02:05'."""
    if sec is None:
        return None
    s = int(round(sec))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search for images.")
    parser.add_argument("--index", type=str, default="faiss_index.faiss", help="Path to the FAISS index.")
    parser.add_argument("--meta", type=str, default="metadata.json", help="Path to the metadata JSON.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Text query to search for.")
    group.add_argument("--image", type=str, help="Path to an image to search with.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to return.")
    parser.add_argument("--source", type=str, default=None,
                        help="Restrict to a video (path or filename), or 'images' / 'videos'.")
    parser.add_argument("--sort", choices=["score", "time"], default="score")

    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Load FAISS index
    index = faiss.read_index(args.index)

    # Load metadata
    with open(args.meta, 'r') as f:
        metadata = json.load(f)

    # Load CLIP model
    clip_model_name = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    clip_model.eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

    if args.text:
        results = search_by_text(args.text, index, metadata, clip_model, clip_processor, device, args.top_k,
                                 source_filter=args.source, sort_by=args.sort)
    elif args.image:
        results = search_by_image(args.image, index, metadata, clip_model, clip_processor, device, args.top_k,
                                  source_filter=args.source, sort_by=args.sort)

    for r in results:
        if r.get("timestamp_sec") is not None:
            r["timestamp"] = format_timestamp(r["timestamp_sec"])
    print(json.dumps(results, indent=4))
