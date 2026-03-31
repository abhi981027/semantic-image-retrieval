import os
import shutil
import json
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io

from ingest import ingest_single_image
from search import search_by_text, search_by_image
import faiss
import torch
from transformers import CLIPProcessor, CLIPModel

app = FastAPI()

# Configuration
IMAGES_DIR = "images"
INDEX_PATH = "faiss_index.faiss"
METADATA_PATH = "metadata.json"
STATIC_DIR = "static"
MIN_SIMILARITY = 0.2  # filter weak matches when using cosine similarity (threshold)
FETCH_K = 100         # fetch more candidates, then re-rank/diversify
SCORE_TIE_EPS = 0.02  # prefer global over object if scores are very close

# Ensure directories exist
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "uploads"), exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# We load models on-demand for search to save memory.
# The index and metadata are loaded once at startup and reloaded after ingestion.
index = None
metadata = {}
clip_model = None
clip_processor = None
device = "cpu"

def load_search_models():
    """Loads CLIP model and processor for search."""
    global clip_model, clip_processor, device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model_name = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    print("CLIP model loaded for search.")

def unload_search_models():
    """Unloads models to free up memory."""
    global clip_model, clip_processor
    del clip_model
    del clip_processor
    clip_model = None
    clip_processor = None
    if 'device' in globals() and device == 'cuda':
        torch.cuda.empty_cache()
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


@app.get("/")
def read_root():
    return FileResponse(os.path.join(STATIC_DIR, 'index.html'))

@app.post("/upload/")
async def upload_images(files: list[UploadFile] = File(...)):
    ingested_files_count = 0
    skipped_files_count = 0
    
    for file in files:
        file_path = os.path.join(IMAGES_DIR, file.filename)
        
        # Check if the image is already indexed
        if any(meta.get('image_path') == file_path for meta in metadata.values()):
            skipped_files_count += 1
            continue

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Run ingestion for the new image
        print(f"Starting ingestion for {file.filename}...")
        ingest_single_image(file_path, INDEX_PATH, METADATA_PATH)
        ingested_files_count += 1

    # Reload data only if new files were added
    if ingested_files_count > 0:
        load_index_and_metadata()
    
    message = f"Processing complete. Ingested: {ingested_files_count} new image(s). Skipped: {skipped_files_count} existing image(s)."
    return JSONResponse(content={"message": message})

@app.post("/search/")
async def search(query_text: str = Form(None), query_image: UploadFile = File(None)):
    if index is None:
        raise HTTPException(status_code=404, detail="Index not found. Please upload and ingest images first.")

    results = []
    try:
        load_search_models()
        if query_text:
            results = search_by_text(query_text, index, metadata, clip_model, clip_processor, device, top_k=FETCH_K, min_similarity=MIN_SIMILARITY)
        elif query_image:
            # Save the uploaded query image temporarily to pass its path
            temp_image_path = os.path.join(STATIC_DIR, "uploads", "temp_" + query_image.filename)
            with open(temp_image_path, "wb") as buffer:
                shutil.copyfileobj(query_image.file, buffer)
            
            results = search_by_image(temp_image_path, index, metadata, clip_model, clip_processor, device, top_k=FETCH_K, min_similarity=MIN_SIMILARITY)
            
            # Clean up the temporary file
            os.remove(temp_image_path)
        else:
            raise HTTPException(status_code=400, detail="Please provide a text query or an image.")
    finally:
        unload_search_models()

    
    # Diversify: keep the best (prefer global on ties) per image
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

    # Prepare results for the frontend
    output_results = []
    for res in final_results:
        meta = res['metadata']
        image_path = meta['image_path']
        filename = os.path.basename(image_path)

        # Copy the result image to a static directory to be served if it's not already there
        static_path = os.path.join(STATIC_DIR, "uploads", filename)
        if not os.path.exists(static_path):
            shutil.copy(image_path, static_path)

        output_results.append({
            "score": res['score'],
            "type": meta['type'],
            "box": meta['box'],
            "image_url": os.path.join("/static", "uploads", filename).replace("\\", "/")
        })

    return JSONResponse(content={"results": output_results})
