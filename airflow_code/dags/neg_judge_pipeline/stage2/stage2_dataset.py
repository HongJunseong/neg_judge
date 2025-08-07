import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from .stage2_config import FRAMES_PER_SAMPLE


class Stage2Dataset(Dataset):
    """
    Stage2 모델 추론용 Dataset
    - 라벨 없이, 단일 비디오 ID에 대한 프레임과 Stage1 logits만 입력
    - 반환: 프레임들, PF logits, 비디오 ID
    """

    def __init__(self,
                 image_root: str,
                 video_ids: list,
                 pf_logits: torch.Tensor,
                 transform=None,
                 frames_per_sample: int = FRAMES_PER_SAMPLE,
                 random_sampling: bool = False):

        self.image_root = image_root
        self.transform = transform
        self.frames_per_sample = frames_per_sample
        self.random_sampling = random_sampling

        self.video_frames = {}  # {video_id: [frame_path, ...]}
        self.valid_ids = []
        self.valid_logits = []

        for i, vid in enumerate(video_ids):
            folder = os.path.join(image_root, vid)
            if not os.path.isdir(folder):
                continue
            frames = sorted([
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.endswith(('.jpg', '.png'))
            ])
            if len(frames) >= frames_per_sample:
                self.video_frames[vid] = frames
                self.valid_ids.append(vid)
                self.valid_logits.append(pf_logits[i])

        print(f"[Stage2Dataset] Loaded {len(self.valid_ids)} valid videos")

    def __len__(self):
        return len(self.valid_ids)

    def __getitem__(self, idx):
        vid = self.valid_ids[idx]
        frames = self.video_frames[vid]

        # Temporal Sampling
        if self.random_sampling:
            idxs = np.sort(np.random.choice(len(frames), self.frames_per_sample, replace=False))
        else:
            idxs = np.linspace(0, len(frames) - 1, self.frames_per_sample, dtype=int)

        selected_paths = [frames[i] for i in idxs]
        images = []
        for path in selected_paths:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            images.append(img)

        pf_logit = self.valid_logits[idx]

        return images, pf_logit, vid
