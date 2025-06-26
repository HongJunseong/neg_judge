import os
import shutil
import pandas as pd
import glob

# 설정값
TRAIN_CSV = "train_split_labels.csv"
VAL_CSV   = "val_split_labels.csv"
TEST_CSV  = "test_split_labels.csv"

VIDEO_DATA_ROOT = "video_data"         # 원본 영상 파일들이 들어있는 폴더 (하위 폴더 포함)
OUTPUT_ROOT = "tsn_dataset"            # 최종 분할 결과가 저장될 폴더
TRAIN_DIR = os.path.join(OUTPUT_ROOT, "train")
VAL_DIR   = os.path.join(OUTPUT_ROOT, "val")
TEST_DIR  = os.path.join(OUTPUT_ROOT, "test")

VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')
SEED = 42  # 분할시 사용한 seed와 동일하게 맞추면 재현 가능

# CSV 파일에서 video_id 목록 읽기
def load_video_ids(csv_file):
    df = pd.read_csv(csv_file)
    # CSV 파일에는 최소한 video_id 컬럼이 있어야 함.
    return df["video_id"].unique()

train_ids = load_video_ids(TRAIN_CSV)
val_ids   = load_video_ids(VAL_CSV)
test_ids  = load_video_ids(TEST_CSV)

print(f"CSV에서 읽은 video_id 개수: Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

# video_data 폴더 내의 모든 영상 파일을 재귀적으로 수집하여, {video_id: full_path} 딕셔너리 생성
video_dict = {}
for root, dirs, files in os.walk(VIDEO_DATA_ROOT):
    for file in files:
        if file.lower().endswith(VIDEO_EXTENSIONS):
            full_path = os.path.join(root, file)
            video_id = os.path.splitext(file)[0]  # 확장자 제거한 파일명
            # 동일 video_id가 여러 개 있을 경우, 먼저 발견한 파일만 사용 (필요시 로직 수정)
            if video_id not in video_dict:
                video_dict[video_id] = full_path

print(f"총 {len(video_dict)}개의 영상 파일을 video_data에서 찾았습니다.")

# 출력 폴더 생성: tsn_dataset/train, tsn_dataset/val, tsn_dataset/test
for split in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    os.makedirs(split, exist_ok=True)

# 각 video_id에 해당하는 영상 파일을 해당 split 폴더로 복사하는 함수
def copy_video_files(video_ids, target_dir):
    for vid in video_ids:
        if vid in video_dict:
            src = video_dict[vid]
            filename = os.path.basename(src)
            dst = os.path.join(target_dir, filename)
            shutil.copy2(src, dst)
            print(f"Copied {src} -> {dst}")
        else:
            print(f"⚠️ 영상 누락: {vid}")

# 각 분할별로 복사 실행
copy_video_files(train_ids, TRAIN_DIR)
copy_video_files(val_ids, VAL_DIR)
copy_video_files(test_ids, TEST_DIR)

print(f"✅ 영상 복사 완료: {TRAIN_DIR}, {VAL_DIR}, {TEST_DIR}")
