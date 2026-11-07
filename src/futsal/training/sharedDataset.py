import hashlib
import math
from pathlib import Path

import cv2
import numpy as np
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils.patches import imread

from futsal.annotations.detectionGeometry import renderYoloLabels


class SharedDetectionDataset(YOLODataset):
    def __init__(self, rows, **kwargs):
        self.sharedRows = sorted(rows, key=lambda row: Path(row["imagePath"]).name if row.get("origin") == "synthetic"
                                 else hashlib.sha256(row["recordId"].encode()).hexdigest())
        self.byRecord = {row["recordId"]: row for row in rows}
        if len(self.byRecord) != len(rows) or not rows:
            raise ValueError("Shared native dataset needs distinct logical records and a nonempty split.")
        if kwargs.get("cache") or kwargs.get("single_cls") or kwargs.get("classes"):
            raise ValueError("Shared native reader requires unmodified classes and disabled image caching.")
        super().__init__(img_path="sharedMedia", **kwargs)

    def get_img_files(self, img_path):
        return [row["resolvedImage"] for row in self.sharedRows]

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample["recordId"] = self.labels[index]["recordId"]
        sample["assetId"] = self.labels[index]["assetId"]
        return sample

    def get_labels(self):
        labels = []
        for row in self.sharedRows:
            content = renderYoloLabels(row["targets"])
            values = np.array([list(map(float, line.split())) for line in content.splitlines()], dtype=np.float32).reshape(-1, 5)
            width, height = row["dimensions"]
            labels.append({"im_file": row["resolvedImage"], "shape": (height, width),
                "recordId": row["recordId"], "assetId": row["imageSha256"],
                "cls": values[:, :1], "bboxes": values[:, 1:], "segments": [],
                "keypoints": None, "normalized": True, "bbox_format": "xywh"})
        return labels

    def load_image(self, i, rect_mode=True, resize_short=False):
        if self.ims[i] is not None:
            return self.ims[i], self.im_hw0[i], self.im_hw[i]
        path = self.im_files[i]
        if Path(path).suffix.lower() in {".jpg", ".jpeg"}:
            with Path(path).open("rb") as stream:
                stream.seek(-2, 2)
                if stream.read() != b"\xff\xd9":
                    raise ValueError(f"Shared JPEG is incomplete; automatic repair is forbidden: {path}.")
        image = imread(path, flags=self.cv2_flag)
        if image is None:
            raise ValueError(f"Cannot decode shared image: {path}.")
        height, width = image.shape[:2]
        row = self.byRecord[self.labels[i]["recordId"]]
        if [width, height] != row["dimensions"]:
            raise ValueError(f"Shared image dimensions changed: {row['recordId']}.")
        shape = tuple(self.imgsz) if isinstance(self.imgsz, (tuple, list)) else (self.imgsz, self.imgsz)
        size = max(shape)
        if rect_mode:
            ratio = size / (min(height, width) if resize_short else max(height, width))
            if ratio != 1:
                target = ((math.ceil(width * ratio), size) if height < width else (size, math.ceil(height * ratio))) if resize_short else (
                    min(math.ceil(width * ratio), size), min(math.ceil(height * ratio), size))
                image = cv2.resize(image, target, interpolation=cv2.INTER_LINEAR)
        elif (height, width) != shape:
            image = cv2.resize(image, shape[::-1], interpolation=cv2.INTER_LINEAR)
        if image.ndim == 2:
            image = image[..., None]
        if self.augment:
            self.ims[i], self.im_hw0[i], self.im_hw[i] = image, (height, width), image.shape[:2]
            self.buffer.append(i)
            if 1 < len(self.buffer) >= self.max_buffer_length:
                index = self.buffer.pop(0)
                self.ims[index], self.im_hw0[index], self.im_hw[index] = None, None, None
        return image, (height, width), image.shape[:2]
