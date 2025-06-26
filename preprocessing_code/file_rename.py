import os
import glob
import re

root = 'data/val'
for video_dir in os.listdir(root):
    video_path = os.path.join(root, video_dir)
    if os.path.isdir(video_path):
        # 파일 이름 패턴: video_id.mp4_frame_*.jpg
        pattern = os.path.join(video_path, f"{video_dir}.mp4_frame_*.jpg")
        files = glob.glob(pattern)
        # 파일 이름에서 frame 번호를 추출하여 정렬
        files_sorted = sorted(files, key=lambda x: int(re.search(r'frame_(\d+)', x).group(1)))
        for i, file in enumerate(files_sorted, 1):
            new_name = os.path.join(video_path, f"img_{i:05d}.jpg")
            os.rename(file, new_name)
            print(f"Renamed {file} -> {new_name}")
