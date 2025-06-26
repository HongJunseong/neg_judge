import os
import pandas as pd

def get_video_filenames(folder_path, video_extensions=('.mp4', '.avi', '.mov', '.mkv')):
    """
    지정한 폴더 내의 파일명을 모두 소문자로 변환하고 좌우 공백 제거 후 set으로 반환합니다.
    """
    filenames = set()
    for file in os.listdir(folder_path):
        if file.lower().endswith(video_extensions):
            filenames.add(file.lower().strip())
    return filenames

def remove_extension(filename):
    """
    파일명에서 확장자를 제거하고 반환합니다.
    """
    return os.path.splitext(filename)[0]

# CSV 파일 읽기
csv_path = "train_data_grouped_with_class.csv"
df = pd.read_csv(csv_path)

# CSV의 영상 식별자 컬럼 이름 (CSV에 영상 파일명이 저장되어 있다고 가정)
video_id_col = "video_id"

# CSV에서 video_id 컬럼에서 파일명만 추출하고 소문자, 좌우 공백 제거
df['video_filename'] = df[video_id_col].apply(lambda x: os.path.basename(str(x)).lower().strip())
# 확장자 제외한 파일명 (비교를 위해)
df['video_filename_noext'] = df['video_filename'].apply(remove_extension)

print("CSV video_filename sample:")
print(df['video_filename'].head())
print("CSV video_filename_noext sample:")
print(df['video_filename_noext'].head())

# tsn_dataset의 각 split 폴더 경로
base_folder = "tsn_dataset"
splits = ["train", "val", "test"]

# 각 split 폴더에 있는 파일명을 수집하고, 확장자 제거한 이름도 함께 저장
split_video_ids = {}
split_video_ids_noext = {}
for split in splits:
    folder = os.path.join(base_folder, split)
    filenames = get_video_filenames(folder)
    split_video_ids[split] = filenames
    split_video_ids_noext[split] = set([remove_extension(x) for x in filenames])
    print(f"{split} 폴더 내 파일 (with ext): {split_video_ids[split]}")
    print(f"{split} 폴더 내 파일 (no ext): {split_video_ids_noext[split]}")

# CSV 데이터에서 각 split에 해당하는 행만 필터링
split_dfs = {}
for split in splits:
    # 우선 확장자를 포함한 비교
    matched_df = df[df['video_filename'].isin(split_video_ids[split])]
    if len(matched_df) == 0:
        # 없으면 확장자 제외한 이름으로 비교
        matched_df = df[df['video_filename_noext'].isin(split_video_ids_noext[split])]
    split_dfs[split] = matched_df
    print(f"CSV 기준 {split} 데이터 개수: {len(matched_df)}")

# 생성할 정보별 컬럼과 txt 파일 접미사 매핑
info_map = {
    "accident_place": "accident_place",
    "accident_place_feature": "accident_place_feature",
    "vehicle_a_progress_info": "vehicle_a_progress_info",
    "vehicle_b_progress_info": "vehicle_b_progress_info"
}

# 각 split별로, 각 정보 컬럼에 대해 txt 파일 생성 (한 줄: "파일명 정보값")
for split, split_df in split_dfs.items():
    for col, suffix in info_map.items():
        txt_filename = f"{split}_{suffix}.txt"
        with open(txt_filename, "w") as f:
            for idx, row in split_df.iterrows():
                f.write(f"{row['video_filename']} {row[col]}\n")
        print(f"{txt_filename} 생성 완료. ({len(split_df)} 항목)")

print("모든 txt 파일 생성 완료.")
