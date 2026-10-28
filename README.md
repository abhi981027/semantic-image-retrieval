# Semantic Image Retrieval System

A powerful image search engine that uses state-of-the-art AI models (SAM + CLIP + FAISS) to enable semantic search through images using text queries or image-based queries. The system can detect and match both entire images and specific objects within images.

## Features

- **Text-to-Image Search**: Find images using natural language descriptions
- **Image-to-Image Search**: Upload an image to find visually similar images
- **Object Detection & Matching**: Automatically detect objects in images and match specific regions
- **Semantic Understanding**: Goes beyond keyword matching using CLIP embeddings
- **Fast Retrieval**: FAISS-powered vector search for instant results
- **Modern Web UI**: Clean, responsive interface with drag-and-drop support
- **Batch Upload**: Process multiple images at once
- **Visual Bounding Boxes**: See exactly which objects matched your query

## Architecture

### Components

1. **Frontend**: Modern HTML/CSS/JavaScript interface
2. **Backend**: FastAPI server handling uploads and search requests
3. **AI Models**:
   - **SAM (Segment Anything Model)**: Class-agnostic object segmentation
   - **CLIP**: Vision-language model for semantic embeddings
4. **Vector Database**: FAISS for efficient similarity search
5. **Storage**: Local file system for images and metadata

### Data Flow

#### Ingestion Pipeline (Upload)

```
Image Upload → Global CLIP Embedding → SAM Segmentation → 
Object Cropping → Object CLIP Embeddings → FAISS Index + Metadata
```

#### Search Pipeline (Query)

```
Text/Image Query → CLIP Embedding → FAISS Search → 
Metadata Lookup → Result Ranking → Display Results
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM recommended
- Git

### Installation

1. **Clone the repository**

   ```powershell
   git clone https://github.com/Prit44421/semantic-image-retrieval.git
   cd semantic-image-retrieval
   ```
2. **Create a virtual environment**

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. **Install dependencies**

   ```powershell
   pip install -r requirements.txt
   ```
4. **Download SAM model weights** (if not already present)

   The SAM checkpoint `sam_vit_b_01ec64.pth` should be in the root directory. If missing, download it:

   ```powershell
   # Download from https://github.com/facebookresearch/segment-anything#model-checkpoints
   # Place sam_vit_b_01ec64.pth in the project root
   ```

### Quick Start

1. **Start the server**

   ```powershell
   uvicorn app.main:app --reload
   ```
2. **Open your browser**

   Navigate to `http://localhost:8000`
3. **Upload images**

   Use the "Upload Images" section to add images to your database
4. **Search**

   Try searching with text like "a cat sitting" or upload a query image

## 📁 Project Structure

```
Image_retrival/
├── app/
│   ├── main.py              # FastAPI application
│   └── __pycache__/
├── static/
│   ├── index.html           # Web interface
│   ├── app.js               # Frontend JavaScript
│   ├── styles.css           # Styling
│   └── uploads/             # Served result images
├── images/                  # Stored uploaded images
├── ingest.py               # Image ingestion pipeline
├── search.py               # Search pipeline
├── faiss_index.faiss       # FAISS vector index
├── metadata.json           # Image metadata mapping
├── requirements.txt        # Python dependencies
├── sam_vit_b_01ec64.pth   # SAM model checkpoint
└── README.md              # This file
```

## 🔧 Configuration

Key configuration variables in `app/main.py`:

```python
IMAGES_DIR = "images"              # Image storage directory
INDEX_PATH = "faiss_index.faiss"   # FAISS index file
METADATA_PATH = "metadata.json"    # Metadata file
MIN_SIMILARITY = 0.2               # Minimum similarity threshold
FETCH_K = 100                      # Number of candidates to fetch
SCORE_TIE_EPS = 0.02              # Tie-breaking epsilon
```

## 💡 How It Works

### 1. Image Ingestion

When you upload an image:

1. **Global Embedding**: The entire image is encoded using CLIP
2. **Segmentation**: SAM detects all objects/regions in the image
3. **Object Embeddings**: Each detected region is cropped and encoded with CLIP
4. **Indexing**: All embeddings are added to the FAISS index
5. **Metadata**: Mappings between index IDs and image paths/bounding boxes are stored

### 2. Search

When you search:

1. **Query Encoding**: Your text/image is converted to a CLIP embedding
2. **Vector Search**: FAISS finds the most similar embeddings
3. **Ranking**: Results are ranked by similarity score
4. **Deduplication**: Best match per image is selected
5. **Display**: Images with bounding boxes (for object matches) are shown

### 3. Why SAM + CLIP?

- **YOLO limitation**: YOLO only detects ~80 predefined classes (person, car, dog, etc.)
- **SAM advantage**: Detects ANY object, even if never seen during training (class-agnostic)
- **CLIP advantage**: Understands semantic relationships between text and images
- **Together**: Unlimited object detection + semantic understanding = powerful search

## 📊 Metadata Format

Each entry in `metadata.json` maps a FAISS index ID to image information:

```json
{
  "0": {
    "image_path": "images/photo.jpg",
    "type": "global",
    "box": null
  },
  "1": {
    "image_path": "images/photo.jpg",
    "type": "object",
    "box": [120, 50, 300, 400]
  }
}
```

- **type**: `global` (whole image) or `object` (detected region)
- **box**: Bounding box coordinates `[x1, y1, x2, y2]` for objects

Entries that come from a video keyframe (Phase 2) carry three extra fields. Image-only
entries are unchanged, so Phase 1 indexes keep working.

```json
{
  "42": {
    "image_path": "frames/surveillance/frame_001050.png",
    "type": "object",
    "box": [400, 200, 800, 600],
    "source_video": "videos/surveillance.mp4",
    "frame_number": 1050,
    "timestamp_sec": 42.0
  }
}
```

## 🎬 Phase 2: Video Retrieval

Upload a video, then search it with text or an image and get back timestamped keyframes
with bounding boxes ("Source: surveillance.mp4 @ 0:42").

### Pipeline

1. **Frame sampling** (`video_ingest.py`): OpenCV reads the video and samples frames at
   1 fps for clips up to 2 minutes and 0.5 fps for longer ones (configurable).
2. **Keyframe selection**: each sampled frame is compared with the last kept keyframe and
   dropped if similarity exceeds the threshold (default 0.95).
   - `histogram`: HSV colour-histogram correlation. Fast, needs no model.
   - `clip`: cosine similarity of CLIP embeddings. More semantic, slower.
   - A keyframe is forced every `max_gap_sec` (default 10 s) so slow scene changes
     (e.g. dashcam footage) are still covered.
3. **Ingestion**: keyframes go through the same SAM → NMS → CLIP pipeline as images and are
   appended to the index with `source_video`, `frame_number` and `timestamp_sec`.
4. **Search**: results include the video fields; you can filter to one video (or to images
   or videos only) and order results by timeline.

### Usage

```bash
# Web UI: "Upload Video" section (runs in the background and shows progress)
uvicorn app.main:app --reload

# Command line
python video_ingest.py --video videos/clip.mp4                 # extract + index
python video_ingest.py --video videos/clip.mp4 --extract-only  # keyframes only
python search.py --text "a red car" --source clip.mp4 --sort time
```

### API

| Endpoint | Description |
|---|---|
| `POST /upload-video/` | Form fields: `file`, `method` (`histogram`/`clip`), `fps` (0 = auto), `similarity_threshold`, `max_gap_sec` (0 = off), `max_keyframes` (0 = no cap). Returns a `job_id`. |
| `GET /video-status/{job_id}` | Stage (`extracting` → `ingesting` → `done`), progress and timing stats |
| `GET /videos/` | Indexed videos and their keyframe counts |
| `POST /search/` | Now also accepts `source_filter` (video path/filename, `images` or `videos`) and `sort_by` (`score`/`time`). Video results include `video_name`, `timestamp`, `timestamp_sec`, `frame_number`. |

### Benchmarking

```bash
# Extraction only (seconds): sampled frames, keyframes, reduction ratio, throughput
python benchmark_video.py --videos videos/*.mp4 --extract-only

# Full run: + SAM/CLIP time per keyframe, search latency, Phase 1 image baseline
python benchmark_video.py --videos videos/*.mp4 --methods histogram clip --images images/ --max-keyframes 20
```

Results are written to `benchmarks/<timestamp>/results.json` and `summary.md`, using a
separate index so the app's index is not modified.

### Apple Silicon notes

- Device order is CUDA → MPS → CPU (`device_utils.get_device()`); force one with
  `EDGESEEK_DEVICE=cpu`.
- If SAM's automatic mask generator fails on MPS, it automatically falls back to CPU for SAM
  (CLIP stays on MPS).

## Advanced Features

### Result Diversification

The system automatically:

- Filters results by minimum similarity threshold
- Deduplicates: shows only the best match per image
- Prefers global matches over object matches when scores are similar
- Ranks by similarity score

### Bounding Box Visualization

When an object match is returned, the UI draws a green bounding box highlighting the matched region in the image.

## 🛠️ Troubleshooting

### Models not loading

- Ensure SAM checkpoint file exists in the root directory
- Check CUDA availability with `torch.cuda.is_available()`

### Out of memory

- Reduce batch size in ingestion
- Use CPU instead of GPU
- Close other applications

### No search results

- Check that images have been uploaded and ingested
- Verify `faiss_index.faiss` and `metadata.json` exist
- Lower `MIN_SIMILARITY` threshold

### Upload fails

- Check file permissions on `images/` directory
- Ensure images are valid formats (JPG, PNG)
- Check available disk space

## 📈 Performance

- **Ingestion**: ~10-30 seconds per image (GPU) / ~30-90 seconds (CPU)
- **Search**: <1 second for most queries
- **Index Size**: ~2KB per embedding (512-dim float32)
- **Scalability**: Tested with 1000+ images

## 📚 Technologies Used

- **FastAPI**: Modern Python web framework
- **PyTorch**: Deep learning framework
- **CLIP**: OpenAI's vision-language model
- **SAM**: Meta's Segment Anything Model
- **FAISS**: Facebook's similarity search library
- **Pillow**: Image processing
- **Uvicorn**: ASGI server

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments

- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) by Meta AI
- [CLIP](https://github.com/openai/CLIP) by OpenAI
- [FAISS](https://github.com/facebookresearch/faiss) by Facebook Research
- [FastAPI](https://fastapi.tiangolo.com/) by Sebastián Ramírez

**Note**: This project is designed for educational and research purposes. For production use, consider additional optimizations, security measures, and scalability improvements.
