import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from stage2_config import FRAMES_PER_SAMPLE

class Stage2Dataset(Dataset):
    """
    2차 모델용 데이터셋:
    - Video frames 불러오기 및 Temporal Augmentation 적용
    - 1차 예측 logits, feature label, A/B 진행 라벨 반환
    """
    def __init__(self,
                 image_root: str,
                 feature_txt: str,
                 a_txt: str,
                 b_txt: str,
                 preds_path: str,
                 transform=None,
                 frames_per_sample: int = FRAMES_PER_SAMPLE,
                 random_sampling: bool = True):
        self.transform = transform
        self.image_root = image_root
        self.frames_per_sample = frames_per_sample
        self.random_sampling = random_sampling

        # 1) 라벨 맵 로드
        self.f_map = {}
        with open(feature_txt, 'r') as f:
            for line in f:
                vid, lbl = line.strip().split()
                self.f_map[vid] = int(lbl)
        self.a_map = {}
        with open(a_txt, 'r') as f:
            for line in f:
                vid, lbl = line.strip().split()
                self.a_map[vid] = int(lbl)
        self.b_map = {}
        with open(b_txt, 'r') as f:
            for line in f:
                vid, lbl = line.strip().split()
                self.b_map[vid] = int(lbl)

        # 2) 비디오별 프레임 경로 저장
        self.video_frames = {}
        for vid in self.f_map:
            if vid in self.a_map and vid in self.b_map:
                folder = os.path.join(image_root, vid)
                if not os.path.isdir(folder):
                    continue
                imgs = sorted([
                    os.path.join(folder, x)
                    for x in os.listdir(folder)
                    if x.endswith(('.jpg', '.png'))
                ])
                if len(imgs) >= frames_per_sample:
                    self.video_frames[vid] = imgs

        # 3) 유효한 sample list
        self.samples = list(self.video_frames.keys())

        # 4) 1차 예측 logits 로드
        data = torch.load(preds_path)
        keys, logits = data['keys'], data['logits']
        self.pf_map = {k: logits[i] for i, k in enumerate(keys)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid = self.samples[idx]
        imgs = self.video_frames[vid]

        # Temporal Augmentation: 랜덤/균일 샘플링
        if self.random_sampling:
            # 랜덤 프레임 인덱스 선택 후 정렬
            idxs = np.sort(
                np.random.choice(
                    len(imgs), self.frames_per_sample, replace=False
                )
            )
        else:
            # 균일 샘플링
            idxs = np.linspace(0, len(imgs) - 1, self.frames_per_sample, dtype=int)
        selected = [imgs[i] for i in idxs]

        # 프레임 로드 및 Transform 적용
        frames = []
        for path in selected:
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        # 라벨 및 예측 logits
        pf_logits = self.pf_map[vid]
        f_lbl = self.f_map[vid]
        a_lbl = self.a_map[vid]
        b_lbl = self.b_map[vid]

        return frames, pf_logits, f_lbl, a_lbl, b_lbl, vid
