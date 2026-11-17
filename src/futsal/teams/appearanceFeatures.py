from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np

from futsal.inventory.catalog import hashFile

siglipFiles = ("config.json", "model.safetensors", "preprocessor_config.json")


def colorHistogram(crop, bins):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1, 2], None, list(bins), [0, 180, 0, 256, 0, 256]).ravel()
    # Square-rooted L1 histograms make Euclidean clustering follow the Hellinger distance.
    return np.sqrt(histogram / max(float(histogram.sum()), 1.0)).astype(np.float32)


def loadSiglip(model, revision, device):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoImageProcessor, SiglipVisionModel

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("Requested CUDA device is not available; no CPU fallback is allowed.")
    snapshot = Path(snapshot_download(model, revision=revision, local_files_only=True,
                                      allow_patterns=list(siglipFiles)))
    processor = AutoImageProcessor.from_pretrained(snapshot)
    network = SiglipVisionModel.from_pretrained(snapshot).to(device).eval()
    evidence = {"model": model, "revision": revision, "device": device,
                "files": {name: hashFile(snapshot / name) for name in siglipFiles},
                "packages": {name: version(name) for name in ("torch", "transformers", "huggingface-hub")}}
    return (processor, network, device), evidence


def siglipEmbeddings(crops, encoder, batchSize):
    import torch

    processor, network, device = encoder
    batches = []
    with torch.inference_mode():
        for start in range(0, len(crops), batchSize):
            images = [cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) for crop in crops[start:start + batchSize]]
            inputs = processor(images=images, return_tensors="pt").to(device)
            pooled = network(**inputs).pooler_output
            batches.append(torch.nn.functional.normalize(pooled, dim=1).float().cpu().numpy())
    if not batches:
        return np.zeros((0, network.config.hidden_size), dtype=np.float32)
    return np.concatenate(batches)
