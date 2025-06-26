import os
import shutil
import pandas as pd

# CSV 파일 경로 (train_split_data.csv)
csv_path = "train_split_labels.csv"
# 대상 폴더 경로 (data2/train)
train_dir = "data2/train"

# CSV 파일에서 valid한 video_id 목록 읽기
df = pd.read_csv(csv_path)
# 문자열로 변환 후 집합으로 저장
valid_video_ids = set(df["video_id"].astype(str).tolist())

print(f"CSV에 포함된 video_id 수: {len(valid_video_ids)}")

# train_dir 내의 모든 폴더 확인
for folder in os.listdir(train_dir):
    folder_path = os.path.join(train_dir, folder)
    if os.path.isdir(folder_path):
        if folder not in valid_video_ids:
            print(f"Deleting folder: {folder_path} (not found in CSV)")
            shutil.rmtree(folder_path)
        else:
            print(f"Keeping folder: {folder_path}")
