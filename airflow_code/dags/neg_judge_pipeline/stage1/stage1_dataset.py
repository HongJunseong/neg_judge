import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class MultiFrameDataset(Dataset):
    def __init__(self, image_root, transform=None, frames_per_sample=7):
        """
        image_root: /path/to/frames/video_id (e.g., /opt/data/frames/video123)
        => video_id 폴더 내부의 프레임만 사용
        """
        self.image_root = image_root
        self.transform = transform
        self.frames_per_sample = frames_per_sample
        self.samples = []  # List of (selected_frame_paths, video_id)

        video_id = os.path.basename(os.path.normpath(image_root))
        if not os.path.isdir(image_root):
            raise ValueError(f"Frame directory not found: {image_root}")

        frame_files = sorted([
            os.path.join(image_root, f)
            for f in os.listdir(image_root)
            if f.endswith(('.jpg', '.png'))
        ])
        if len(frame_files) < self.frames_per_sample:
            raise ValueError(f"Not enough frames in {image_root} (found {len(frame_files)}, required {self.frames_per_sample})")

        indices = np.linspace(0, len(frame_files) - 1, self.frames_per_sample, dtype=int)
        selected = [frame_files[i] for i in indices]
        self.samples.append((selected, video_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, video_id = self.samples[idx]
        images = []
        for path in frame_paths:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            images.append(img)
        return images, video_id
