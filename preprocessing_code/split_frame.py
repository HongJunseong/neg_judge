import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import glob

# 설정
LABEL_CSV         = "train_data_grouped_with_class.csv"
PROCESSED_DATA_DIR= "processed_data"
SOURCE_FRAME_ROOTS= [
    os.path.join(PROCESSED_DATA_DIR, d)
    for d in os.listdir(PROCESSED_DATA_DIR)
    if os.path.isdir(os.path.join(PROCESSED_DATA_DIR, d))
]
TRAIN_DIR = "data2/train"
VAL_DIR   = "data2/val"
TEST_DIR  = "data2/test"

# 최소 샘플 수 설정 (필요에 따라 조정)
MIN_PLACE = 10
MIN_A     = 10
MIN_B     = 10

# CSV 로드
df = pd.read_csv(LABEL_CSV)

# valid_classes 함수 정의
def valid_classes(df, col, min_samples):
    counts = (
        df[["video_id", col]]
        .drop_duplicates(subset=["video_id", col])
        [col]
        .value_counts()
    )
    return set(counts[counts >= min_samples].index)

# 각 valid set 계산
valid_place = valid_classes(df, "accident_place_feature",   MIN_PLACE)
valid_a     = valid_classes(df, "vehicle_a_progress_info",  MIN_A)
valid_b     = valid_classes(df, "vehicle_b_progress_info",  MIN_B)

# 세 기준 모두 만족하는 샘플만 남기기 (소수 클래스 제거)
df_filtered = df[
    df["accident_place_feature"].isin(valid_place) &
    df["vehicle_a_progress_info"].isin(valid_a) &
    df["vehicle_b_progress_info"].isin(valid_b)
].reset_index(drop=True)

print(f"필터링 전 비디오 수: {df['video_id'].nunique()}")
print(f"필터링 후 비디오 수: {df_filtered['video_id'].nunique()}")

# Stratified split by accident_place_feature
unique_ids = df_filtered[["video_id","negligence_class"]].drop_duplicates()

train_val_ids, test_ids = train_test_split(
    unique_ids,
    test_size=0.1,
    stratify=unique_ids["negligence_class"],
    random_state=42
)
train_ids, val_ids = train_test_split(
    train_val_ids,
    test_size=0.1111,  # train:val:test = 80:10:10
    stratify=train_val_ids["negligence_class"],
    random_state=42
)

# split 된 레이블 저장 (옵션)
train_df = df_filtered[df_filtered["video_id"].isin(train_ids["video_id"])]
val_df   = df_filtered[df_filtered["video_id"].isin(val_ids["video_id"])]
test_df  = df_filtered[df_filtered["video_id"].isin(test_ids["video_id"])]

train_df.to_csv("train_split_labels.csv", index=False)
val_df.to_csv("val_split_labels.csv", index=False)
test_df.to_csv("test_split_labels.csv", index=False)
print(f"Split CSV: {len(train_df)} train / {len(val_df)} val / {len(test_df)} test")

# 프레임 복사 함수
def copy_frames_by_video_ids(video_ids, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    for vid in video_ids:
        tgt_dir = os.path.join(target_dir, vid)
        os.makedirs(tgt_dir, exist_ok=True)
        found = False
        for root in SOURCE_FRAME_ROOTS:
            pattern = os.path.join(root, f"{vid}.mp4_frame_*.jpg")
            for path in glob.glob(pattern):
                shutil.copy(path, os.path.join(tgt_dir, os.path.basename(path)))
                found = True
        if not found:
            print(f"⚠️ 누락 프레임: {vid}")

# 프레임 분할 복사 실행
copy_frames_by_video_ids(train_ids["video_id"], TRAIN_DIR)
copy_frames_by_video_ids(val_ids["video_id"],   VAL_DIR)
copy_frames_by_video_ids(test_ids["video_id"],  TEST_DIR)
print(f"✅ 프레임 복사 완료: {TRAIN_DIR}, {VAL_DIR}, {TEST_DIR}")
