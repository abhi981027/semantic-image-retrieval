
import faiss
import json
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import argparse

def search_by_text(query_text, index, metadata, clip_model, clip_processor, device, top_k=50, min_similarity: float | None = None):
    inputs = clip_processor(text=query_text, return_tensors="pt").to(device)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)
    
    query_embedding = text_features.cpu().numpy().astype('float32')
    sims, indices = index.search(query_embedding, top_k)
    
    results = []
    for i in range(top_k):
        idx = indices[0][i]
        result_metadata = metadata.get(str(idx))
        if result_metadata is None:
            continue
        sim = float(sims[0][i])
        if min_similarity is not None and sim < min_similarity:
            continue
        results.append({
            "id": int(idx),
            "score": sim,  # cosine similarity in [-1,1]
            "metadata": result_metadata
        })
    return results

def search_by_image(image_path, index, metadata, clip_model, clip_processor, device, top_k=50, min_similarity: float | None = None):
    image = Image.open(image_path).convert("RGB")
    # Force channels_last to avoid ambiguity on very small images
    inputs = clip_processor(images=image, return_tensors="pt", input_data_format="channels_last").to(device)
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
        image_features = torch.nn.functional.normalize(image_features, dim=-1)
    
    query_embedding = image_features.cpu().numpy().astype('float32')
    sims, indices = index.search(query_embedding, top_k)
    
    results = []
    for i in range(top_k):
        idx = indices[0][i]
        result_metadata = metadata.get(str(idx))
        if result_metadata is None:
            continue
        sim = float(sims[0][i])
        if min_similarity is not None and sim < min_similarity:
            continue
        results.append({
            "id": int(idx),
            "score": sim,  # cosine similarity in [-1,1]
            "metadata": result_metadata
        })
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search for images.")
    parser.add_argument("--index", type=str, default="faiss_index.faiss", help="Path to the FAISS index.")
    parser.add_argument("--meta", type=str, default="metadata.json", help="Path to the metadata JSON.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Text query to search for.")
    group.add_argument("--image", type=str, help="Path to an image to search with.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to return.")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
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
        results = search_by_text(args.text, index, metadata, clip_model, clip_processor, device, args.top_k)
    elif args.image:
        results = search_by_image(args.image, index, metadata, clip_model, clip_processor, device, args.top_k)
    
    print(json.dumps(results, indent=4))
