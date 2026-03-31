import os
import json
import torch
import faiss
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import argparse
from typing import List, Tuple

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
    sam_checkpoint = "sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    sam_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    
    if not os.path.exists(sam_checkpoint):
        import requests
        print(f"Downloading SAM checkpoint from {sam_url}...")
        response = requests.get(sam_url, stream=True)
        with open(sam_checkpoint, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    return sam

def ingest_single_image(image_path, index_path, metadata_path):
    """
    Original function to ingest a single image, loading and unloading models.
    This is kept for completeness but is slower for multiple images.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- SAM PART: Generate Masks ---
    print("Loading SAM model to generate masks...")
    sam = get_sam_model(device)
    # Tune SAM to reduce tiny/unstable masks
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        box_nms_thresh=0.7,
        min_mask_region_area=1000,  # roughly filter small fragments
    )
    
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    masks = mask_generator.generate(image_np)
    
    del sam
    del mask_generator
    if device == 'cuda':
        torch.cuda.empty_cache()
    print("SAM model unloaded.")

    # --- CLIP PART: Generate Embeddings ---
    print("Loading CLIP model to generate embeddings...")
    clip_model_name = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    clip_model.eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

    new_embeddings = []
    new_metadata_entries = {}

    # 1. Global embedding
    img_for_clip = _prepare_pil(image)
    # Force channels_last to avoid ambiguity on tiny images (e.g., 1xN or Nx1)
    inputs = clip_processor(images=img_for_clip, return_tensors="pt", input_data_format="channels_last").to(device)
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
        # L2-normalize for cosine/IP search
        image_features = torch.nn.functional.normalize(image_features, dim=-1)
    
    global_embedding = image_features.cpu().numpy().astype("float32").flatten()
    new_embeddings.append(global_embedding)
    new_metadata_entries[0] = {"image_path": image_path, "type": "global", "box": None}

    # 2. Object embeddings
    # First, filter masks by area and stability, then run NMS to remove dupes
    W, H = image.size
    img_area = W * H
    candidate_boxes: List[List[int]] = []
    candidate_scores: List[float] = []
    for mask_data in masks:
        box_xywh = [int(round(v)) for v in mask_data['bbox']]
        x, y, w, h = box_xywh
        # Basic validity
        if w <= 2 or h <= 2:
            continue
        area = mask_data.get('area', w * h)
        stability = mask_data.get('stability_score', 1.0)
        pred_iou = mask_data.get('predicted_iou', 1.0)
        # Drop tiny or unstable masks (e.g., < 1% of image area)
        if area < 0.1 * img_area:
            continue
        if stability < 0.85:
            continue
        candidate_boxes.append(_xywh_to_xyxy([x, y, w, h]))
        # Prefer higher IoU/stability
        candidate_scores.append(0.7 * pred_iou + 0.3 * stability)

    keep_idx = _nms(candidate_boxes, candidate_scores, iou_thresh=0.8)
    # Keep only the top-N highest scoring boxes to avoid background clutter
    MAX_OBJECTS_PER_IMAGE = 12
    keep_idx = keep_idx[:MAX_OBJECTS_PER_IMAGE]

    for i_keep, k in enumerate(keep_idx):
        x1, y1, x2, y2 = candidate_boxes[k]
        # Crop and sanitize image for CLIP
        cropped_image = image.crop((x1, y1, x2, y2))
        cropped_image = _prepare_pil(cropped_image)

        # Force channels_last to avoid ambiguity on extremely small crops
        inputs = clip_processor(images=cropped_image, return_tensors="pt", input_data_format="channels_last").to(device)
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
            image_features = torch.nn.functional.normalize(image_features, dim=-1)
        
        object_embedding = image_features.cpu().numpy().astype("float32").flatten()
        new_embeddings.append(object_embedding)
        new_metadata_entries[i_keep + 1] = {"image_path": image_path, "type": "object", "box": [int(x1), int(y1), int(x2), int(y2)]}

    del clip_model
    del clip_processor
    if device == 'cuda':
        torch.cuda.empty_cache()
    print("CLIP model unloaded.")

    # --- Update FAISS Index and Metadata ---
    if not new_embeddings:
        print("No embeddings were generated for this image.")
        return

    embedding_dim = new_embeddings[0].shape[0]
    
    if os.path.exists(index_path):
        print("Loading existing FAISS index...")
        index = faiss.read_index(index_path)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        start_id = max(map(int, metadata.keys())) + 1 if metadata else 0
    else:
        print("Creating new FAISS index (Inner Product / cosine)...")
        index = faiss.IndexFlatIP(embedding_dim)
        metadata = {}
        start_id = 0

    embeddings_np = np.array(new_embeddings, dtype='float32')
    index.add(embeddings_np)
    
    for i, meta in new_metadata_entries.items():
        metadata[start_id + i] = meta

    faiss.write_index(index, index_path)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Ingestion complete for {image_path}. Index and metadata updated.")

def ingest_images_directory(images_dir, index_path, metadata_path):
    """
    Processes all images in a directory, loading models only once.
    This clears any existing index/metadata files.
    """
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print("No images found in the specified directory.")
        return
        
    # For a full directory ingestion, clear old files
    if os.path.exists(index_path):
        os.remove(index_path)
    if os.path.exists(metadata_path):
        os.remove(metadata_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 1. Load models ONCE ---
    print("Loading SAM model...")
    sam = get_sam_model(device)
    
    print("Using faster SAM parameters (points_per_side=16, min_mask_region_area=5000)")
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=16, 
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        box_nms_thresh=0.7,
        min_mask_region_area=5000,
    )
    
    print("Loading CLIP model...")
    clip_model_name = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    clip_model.eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    print(device)

    all_embeddings = []
    all_metadata = {}
    current_id = 0
    embedding_dim = -1

    # --- 2. Process all images in a loop ---
    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        print(f"\n--- Processing {image_file} ---")

        try:
            image = Image.open(image_path).convert("RGB")
            image_np = np.array(image)
            W, H = image.size
            img_area = W * H

            # --- SAM Part ---
            masks = mask_generator.generate(image_np)

            # --- CLIP Part: Global Embedding ---
            img_for_clip = _prepare_pil(image)
            inputs = clip_processor(images=img_for_clip, return_tensors="pt", input_data_format="channels_last").to(device)
            with torch.no_grad():
                image_features = clip_model.get_image_features(**inputs)
                image_features = torch.nn.functional.normalize(image_features, dim=-1)
            
            global_embedding = image_features.cpu().numpy().astype("float32").flatten()
            
            if embedding_dim == -1:
                embedding_dim = global_embedding.shape[0]

            all_embeddings.append(global_embedding)
            all_metadata[current_id] = {"image_path": image_path, "type": "global", "box": None}
            current_id += 1

            # --- CLIP Part: Object Candidates ---
            candidate_boxes: List[List[int]] = []
            candidate_scores: List[float] = []
            
            for mask_data in masks:
                box_xywh = [int(round(v)) for v in mask_data['bbox']]
                x, y, w, h = box_xywh
                if w <= 2 or h <= 2:
                    continue
                area = mask_data.get('area', w * h)
                stability = mask_data.get('stability_score', 1.0)
                pred_iou = mask_data.get('predicted_iou', 1.0)
                
                # Keep stability/area filters
                if area < 0.1 * img_area: # Use a relative filter
                    continue
                if stability < 0.85:
                    continue
                
                candidate_boxes.append(_xywh_to_xyxy([x, y, w, h]))
                candidate_scores.append(0.7 * pred_iou + 0.3 * stability)

            keep_idx = _nms(candidate_boxes, candidate_scores, iou_thresh=0.8)
            MAX_OBJECTS_PER_IMAGE = 12
            # NMS returns items sorted by score
            keep_idx = keep_idx[:MAX_OBJECTS_PER_IMAGE]

            # *** MODIFICATION: Batch process object embeddings ***
            cropped_images_for_clip = []
            final_boxes = []

            for k in keep_idx:
                x1, y1, x2, y2 = candidate_boxes[k]
                cropped_image = image.crop((x1, y1, x2, y2))
                cropped_images_for_clip.append(_prepare_pil(cropped_image))
                final_boxes.append([int(x1), int(y1), int(x2), int(y2)])
            
            if cropped_images_for_clip:
                # Process all cropped images in one single batch
                inputs = clip_processor(
                    images=cropped_images_for_clip, 
                    return_tensors="pt", 
                    padding=True, # Pad to same size for batching
                    truncation=True, # Truncate if needed
                    input_data_format="channels_last"
                ).to(device)
                
                with torch.no_grad():
                    image_features_batch = clip_model.get_image_features(**inputs)
                    image_features_batch = torch.nn.functional.normalize(image_features_batch, dim=-1)
                
                object_embeddings_batch = image_features_batch.cpu().numpy().astype("float32")
                
                # Add batch results to our main lists
                for i, embedding in enumerate(object_embeddings_batch):
                    all_embeddings.append(embedding)
                    all_metadata[current_id] = {
                        "image_path": image_path, 
                        "type": "object", 
                        "box": final_boxes[i]
                    }
                    current_id += 1
            
            print(f"Found {len(cropped_images_for_clip) + 1} total embeddings for {image_file}.")

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    # --- 3. Unload models ONCE ---
    del sam, mask_generator, clip_model, clip_processor
    if device == 'cuda':
        torch.cuda.empty_cache()
    print("\nModels unloaded.")

    if not all_embeddings:
        print("No embeddings were generated for the entire directory.")
        return

    print(f"Creating FAISS index for {len(all_embeddings)} total embeddings...")
    embeddings_np = np.array(all_embeddings, dtype='float32')
    
    # Ensure all embeddings have the same dimension
    if embeddings_np.ndim != 2 or embeddings_np.shape[1] != embedding_dim:
        print(f"Error: Embedding dimensions are inconsistent. Expected {embedding_dim}, but got shape {embeddings_np.shape}")
        return

    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings_np)
    
    faiss.write_index(index, index_path)
    with open(metadata_path, 'w') as f:
        json.dump(all_metadata, f, indent=4)
    
    print(f"Ingestion complete. Processed {len(image_files)} images.")
    print(f"Index saved to: {index_path}")
    print(f"Metadata saved to: {metadata_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest images and create embeddings.")
    parser.add_argument("--images", type=str, default="images/", help="Directory with images to ingest.")
    parser.add_argument("--index", type=str, default="faiss_index.faiss", help="Path to save the FAISS index.")
    parser.add_argument("--meta", type=str, default="metadata.json", help="Path to save the metadata JSON.")
    args = parser.parse_args()

    ingest_images_directory(args.images, args.index, args.meta)
