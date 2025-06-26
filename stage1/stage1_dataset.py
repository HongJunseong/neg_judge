import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from config import *

class MultiFrameDataset(Dataset):
    def __init__(self, image_root, label_txt, transform=None, frames_per_sample=FRAMES_PER_SAMPLE):
        self.image_root = image_root
        self.transform = transform
        self.frames_per_sample = frames_per_sample
        self.samples = []  # List of tuples: (frame_paths, label, video_id)

        with open(label_txt, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            video_id, label = parts[0], int(parts[1])
            video_path = os.path.join(image_root, video_id)
            if not os.path.isdir(video_path):
                continue

            frame_files = sorted([
                os.path.join(video_path, f)
                for f in os.listdir(video_path)
                if f.endswith(('.jpg', '.png'))
            ])
            if len(frame_files) < self.frames_per_sample:
                continue

            indices = np.linspace(0, len(frame_files) - 1, self.frames_per_sample, dtype=int)
            selected = [frame_files[i] for i in indices]
            self.samples.append((selected, label, video_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label, video_id = self.samples[idx]
        images = []
        for p in frame_paths:
            img = Image.open(p).convert("RGB")
            if self.transform:
                img = self.transform(img)
            images.append(img)
        # images: list of tensors [C,H,W]
        return images, label, video_id
