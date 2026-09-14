"""Standalone FastAPI model server for local PyTorch video embedding extraction.

Provides local feature extraction fallback for environments without Twelve Labs API keys.
"""

from __future__ import annotations

import logging
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Global model state
_model = None
_device = None
_transform = None


def load_local_model(device_str: Optional[str] = None):
    """Loads a PyTorch video embedding backbone (MobileNetV3 / EfficientNet / VideoMAE)."""
    global _model, _device, _transform

    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    _device = torch.device(device_str)

    try:
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        weights = MobileNet_V3_Small_Weights.DEFAULT
        base = mobilenet_v3_small(weights=weights)
        base.classifier = torch.nn.Identity()
        _model = base.to(_device).eval()
        _transform = weights.transforms()
        logger.info("Loaded PyTorch MobileNetV3-Small backbone on %s", _device)
    except Exception as exc:
        logger.warning("Falling back to unweighted MobileNetV3: %s", exc)
        from torchvision.models import mobilenet_v3_small
        base = mobilenet_v3_small()
        base.classifier = torch.nn.Identity()
        _model = base.to(_device).eval()
        _transform = T.Compose([
            T.ToPILImage(),
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return _model, _transform


def extract_frames_from_video(video_path: str, num_frames: int = 16) -> np.ndarray:
    """Reads evenly spaced RGB frames from a video file using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0 or not cap.isOpened():
        cap.release()
        raise ValueError(f"Could not open or read video from {video_path}")

    indices = np.linspace(0, max(0, total_frames - 1), num_frames, dtype=int)
    frames_dict = {}
    current_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if current_idx in indices:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_dict[current_idx] = rgb
        current_idx += 1
    cap.release()

    if not frames_dict:
        raise ValueError(f"No frames could be extracted from {video_path}")

    # Assemble frames in chronological order, filling any missing with nearest
    collected = []
    last_frame = next(iter(frames_dict.values()))
    for idx in indices:
        if idx in frames_dict:
            last_frame = frames_dict[idx]
        collected.append(last_frame)

    return np.stack(collected, axis=0)  # (16, H, W, 3)


def extract_embedding_from_path(video_path: str) -> np.ndarray:
    """Extracts a temporal average-pooled, L2-normalized embedding vector from a video file."""
    if _model is None:
        load_local_model()

    from PIL import Image
    frames = extract_frames_from_video(video_path, num_frames=16)
    tensor_list = []
    for f in frames:
        pil_img = Image.fromarray(f) if isinstance(f, np.ndarray) else f
        t = _transform(pil_img)
        tensor_list.append(t)

    batch_tensor = torch.stack(tensor_list, dim=0).to(_device)  # (16, 3, 224, 224)

    with torch.no_grad():
        frame_features = _model(batch_tensor)  # (16, 576)
        clip_emb = frame_features.mean(dim=0)   # (576,)
        norm = torch.norm(clip_emb, p=2)
        if norm > 0:
            clip_emb = clip_emb / norm
        emb_np = clip_emb.cpu().numpy()

    return emb_np


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_local_model()
    yield


app = FastAPI(
    title="Local Video Embedding Model Server",
    description="High-precision PyTorch video feature extraction server",
    lifespan=lifespan,
)


class EmbeddingResponse(BaseModel):
    embedding: List[float]
    dim: int
    latency_ms: float


class BatchEmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    count: int
    latency_ms: float


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "device": str(_device),
    }


@app.post("/embed", response_model=EmbeddingResponse)
async def embed(file: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(503, "Model not loaded")

    t0 = time.perf_counter()
    content = await file.read()

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        emb = extract_embedding_from_path(tmp_path)
    except Exception as exc:
        raise HTTPException(400, f"Failed to extract embedding from video: {exc}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    latency = (time.perf_counter() - t0) * 1000
    return EmbeddingResponse(
        embedding=emb.tolist(),
        dim=len(emb),
        latency_ms=latency,
    )


@app.post("/batch-embed", response_model=BatchEmbeddingResponse)
async def batch_embed(files: List[UploadFile] = File(...)):
    if _model is None:
        raise HTTPException(503, "Model not loaded")

    t0 = time.perf_counter()
    embeddings = []

    for f in files:
        content = await f.read()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            emb = extract_embedding_from_path(tmp_path)
            embeddings.append(emb.tolist())
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    latency = (time.perf_counter() - t0) * 1000
    return BatchEmbeddingResponse(
        embeddings=embeddings,
        count=len(embeddings),
        latency_ms=latency,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9877)
