import os
import json
import time
import torch
import faiss
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import argparse
from typing import Callable, Dict, List, Optional, Tuple

from device_utils import get_device, free_device_memory

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

MAX_OBJECTS_PER_IMAGE = 12
MIN_OBJECT_AREA_FRAC = 0.1   # drop masks smaller than 10% of the image area
MIN_STABILITY = 0.85
OBJECT_NMS_IOU = 0.8

# Phase 1 used two SAM settings: a higher-quality one for single uploads and a
# faster one for batch/directory ingestion. Both are kept unchanged.
SAM_PARAMS_QUALITY = dict(
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    box_nms_thresh=0.7,
    min_mask_region_area=1000,
)
SAM_PARAMS_FAST = dict(
    points_per_side=16,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    box_nms_thresh=0.7,
    min_mask_region_area=5000,
)


def _xywh_to_xyxy(box: List[int]) -> List[int]:
    x, y, w, h = box
    return [x, y, x + w, y + h]

def _iou(boxA: List[int], boxB: List[int]) -> float:
    # boxes are [x1,y1,x2,y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = areaA + areaB - interArea
    return float(interArea) / union if union > 0 else 0.0

def _nms(boxes: List[List[int]], scores: List[float], iou_thresh: float = 0.9) -> List[int]:
    # Simple NMS returning kept indices
    if not boxes:
        return []
    idxs = list(range(len(boxes)))
    # sort by score desc
    idxs.sort(key=lambda i: scores[i], reverse=True)
    keep: List[int] = []
    while idxs:
        i = idxs.pop(0)
        keep.append(i)
        idxs = [j for j in idxs if _iou(boxes[i], boxes[j]) < iou_thresh]
    return keep

def _prepare_pil(img: Image.Image) -> Image.Image:
    """Ensure PIL RGB image."""
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.array(img))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

def get_sam_model(device):
    if not os.path.exists(SAM_CHECKPOINT):
        import requests
        print(f"Downloading SAM checkpoint from {SAM_URL}...")
        response = requests.get(SAM_URL, stream=True)
        with open(SAM_CHECKPOINT, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")

    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=device)
    return sam


class SafeMaskGenerator:
    """
    Wraps SamAutomaticMaskGenerator and falls back to CPU if SAM fails on MPS.

    segment_anything's automatic mask generator creates float64 tensors internally,
    which Apple's MPS backend does not support. Rather than guessing, we try the
    requested device first and permanently switch SAM to CPU on the first failure.
    CLIP stays on MPS either way.
    """

    def __init__(self, device: str, sam_params: dict):
        self.device = device
        self.sam_params = sam_params
        self._build(device)

    def _build(self, device: str):
        self.device = device
        self.sam = get_sam_model(device)
        self.generator = SamAutomaticMaskGenerator(self.sam, **self.sam_params)

    def generate(self, image_np: np.ndarray):
        try:
            return self.generator.generate(image_np)
        except (TypeError, RuntimeError, NotImplementedError) as e:
            if self.device == "cpu":
                raise
            print(f"SAM failed on {self.device} ({e}). Falling back to CPU for SAM.")
            del self.generator, self.sam
            free_device_memory(self.device)
            self._build("cpu")
            return self.generator.generate(image_np)


class IngestModels:
    """SAM + CLIP loaded once and reused across many images/frames."""

    def __init__(self, device: Optional[str] = None, sam_params: Optional[dict] = None):
        self.device = device or get_device()
        print(f"Using device: {self.device}")
        print("Loading SAM model...")
        self.mask_generator = SafeMaskGenerator(self.device, sam_params or SAM_PARAMS_FAST)
        print("Loading CLIP model...")
        self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(self.device)
        self.clip_model.eval()
        self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    def unload(self):
        del self.mask_generator, self.clip_model, self.clip_processor
        free_device_memory(self.device)
        print("Models unloaded.")


def clip_image_embeddings(images: List[Image.Image], clip_model, clip_processor, device) -> np.ndarray:
    """Batch CLIP image embeddings, L2-normalised -> (N, 512) float32."""
    inputs = clip_processor(
        images=[_prepare_pil(im) for im in images],
        return_tensors="pt",
        # Force channels_last to avoid ambiguity on tiny images (e.g., 1xN or Nx1)
        input_data_format="channels_last",
    ).to(device)
    with torch.no_grad():
        feats = clip_model.get_image_features(**inputs)
        feats = torch.nn.functional.normalize(feats, dim=-1)
    return feats.cpu().numpy().astype("float32")


def select_object_boxes(masks, img_w: int, img_h: int) -> List[List[int]]:
    """Area/stability filters -> NMS -> top-N boxes in [x1, y1, x2, y2]."""
    img_area = img_w * img_h
    candidate_boxes: List[List[int]] = []
    candidate_scores: List[float] = []
    for mask_data in masks:
        x, y, w, h = [int(round(v)) for v in mask_data['bbox']]
        if w <= 2 or h <= 2:
            continue
        area = mask_data.get('area', w * h)
        stability = mask_data.get('stability_score', 1.0)
        pred_iou = mask_data.get('predicted_iou', 1.0)
        if area < MIN_OBJECT_AREA_FRAC * img_area:
            continue
        if stability < MIN_STABILITY:
            continue
        candidate_boxes.append(_xywh_to_xyxy([x, y, w, h]))
        # Prefer higher IoU/stability
        candidate_scores.append(0.7 * pred_iou + 0.3 * stability)

    keep_idx = _nms(candidate_boxes, candidate_scores, iou_thresh=OBJECT_NMS_IOU)
    keep_idx = keep_idx[:MAX_OBJECTS_PER_IMAGE]  # NMS returns indices sorted by score
    return [[int(v) for v in candidate_boxes[k]] for k in keep_idx]


def embed_image(image: Image.Image, models: IngestModels) -> List[Tuple[np.ndarray, str, Optional[List[int]]]]:
    """
    Returns [(embedding, type, box), ...]: one global entry followed by object entries.
    Global + object crops are encoded by CLIP in a single batch.
    """
    image = _prepare_pil(image)
    masks = models.mask_generator.generate(np.array(image))
    W, H = image.size
    boxes = select_object_boxes(masks, W, H)
    crops = [image.crop(tuple(b)) for b in boxes]
    embs = clip_image_embeddings([image] + crops, models.clip_model, models.clip_processor, models.device)
    out = [(embs[0], "global", None)]
    out += [(embs[i + 1], "object", boxes[i]) for i in range(len(boxes))]
    return out


def _load_index_and_metadata(index_path: str, metadata_path: str, dim: int):
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        if len(metadata) != index.ntotal:
            print(f"WARNING: metadata has {len(metadata)} entries but index has {index.ntotal} vectors.")
    else:
        print("Creating new FAISS index (Inner Product / cosine)...")
        index = faiss.IndexFlatIP(dim)
        metadata = {}
    return index, metadata


def _atomic_write_json(obj, path: str):
    tmp = path + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=4)
    os.replace(tmp, path)


def append_to_index(embeddings: List[np.ndarray], entries: List[dict], index_path: str, metadata_path: str) -> int:
    """Add vectors + metadata. FAISS IDs are positional, so new IDs start at index.ntotal."""
    if not embeddings:
        return 0
    emb = np.asarray(embeddings, dtype='float32')
    index, metadata = _load_index_and_metadata(index_path, metadata_path, emb.shape[1])
    start_id = index.ntotal
    index.add(emb)
    for i, meta in enumerate(entries):
        metadata[str(start_id + i)] = meta
    faiss.write_index(index, index_path)
    _atomic_write_json(metadata, metadata_path)
    return len(entries)


def ingest_image_list(
    image_paths: List[str],
    index_path: str,
    metadata_path: str,
    extra_metadata: Optional[List[Optional[dict]]] = None,
    append: bool = True,
    models: Optional[IngestModels] = None,
    sam_params: Optional[dict] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Dict:
    """
    Core ingestion routine used by image uploads, directory ingestion and video keyframes.

    extra_metadata: optional list aligned with image_paths; each dict is merged into every
        metadata entry (global + objects) generated from that image. Video keyframes use this
        for {source_video, frame_number, timestamp_sec}. Image-only entries simply omit it,
        so the Phase 1 schema {image_path, type, box} is unchanged.
    append: add to the existing index (True) or start a fresh one (False).
    models: pass pre-loaded IngestModels to reuse them; otherwise they're loaded and unloaded here.

    Returns timing stats.
    """
    if not append:
        for p in (index_path, metadata_path):
            if os.path.exists(p):
                os.remove(p)

    own_models = models is None
    if own_models:
        models = IngestModels(sam_params=sam_params)

    all_embeddings: List[np.ndarray] = []
    all_entries: List[dict] = []
    per_image_sec: List[float] = []
    failed = 0
    n = len(image_paths)

    try:
        for i, image_path in enumerate(image_paths):
            t0 = time.perf_counter()
            try:
                image = Image.open(image_path).convert("RGB")
                results = embed_image(image, models)
                extra = (extra_metadata[i] if extra_metadata else None) or {}
                for emb, etype, box in results:
                    all_embeddings.append(emb)
                    all_entries.append({"image_path": image_path, "type": etype, "box": box, **extra})
                print(f"[{i + 1}/{n}] {os.path.basename(image_path)}: {len(results)} embeddings "
                      f"({time.perf_counter() - t0:.1f}s)")
            except Exception as e:
                failed += 1
                print(f"Error processing {image_path}: {e}")
            per_image_sec.append(time.perf_counter() - t0)
            if progress_cb:
                progress_cb(i + 1, n)
    finally:
        if own_models:
            models.unload()

    added = append_to_index(all_embeddings, all_entries, index_path, metadata_path)
    if added == 0:
        print("No embeddings were generated.")
    return {
        "images": n,
        "failed": failed,
        "embeddings_added": added,
        "total_sec": float(sum(per_image_sec)),
        "mean_sec_per_image": float(np.mean(per_image_sec)) if per_image_sec else 0.0,
        "per_image_sec": per_image_sec,
    }


def ingest_single_image(image_path, index_path, metadata_path, extra_metadata: Optional[dict] = None):
    """
    Ingest one image into the existing index (Phase 1 behaviour: higher-quality SAM settings,
    models loaded and unloaded for this call).
    """
    return ingest_image_list(
        [image_path], index_path, metadata_path,
        extra_metadata=[extra_metadata] if extra_metadata else None,
        append=True, sam_params=SAM_PARAMS_QUALITY,
    )


def ingest_images_directory(images_dir, index_path, metadata_path):
    """
    Processes all images in a directory, loading models only once.
    This clears any existing index/metadata files (Phase 1 behaviour).
    """
    image_files = sorted(f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    if not image_files:
        print("No images found in the specified directory.")
        return None
    paths = [os.path.join(images_dir, f) for f in image_files]
    stats = ingest_image_list(paths, index_path, metadata_path, append=False, sam_params=SAM_PARAMS_FAST)
    print(f"Ingestion complete. Processed {len(image_files)} images.")
    print(f"Index saved to: {index_path}")
    print(f"Metadata saved to: {metadata_path}")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest images and create embeddings.")
    parser.add_argument("--images", type=str, default="images/", help="Directory with images to ingest.")
    parser.add_argument("--index", type=str, default="faiss_index.faiss", help="Path to save the FAISS index.")
    parser.add_argument("--meta", type=str, default="metadata.json", help="Path to save the metadata JSON.")
    args = parser.parse_args()

    ingest_images_directory(args.images, args.index, args.meta)
