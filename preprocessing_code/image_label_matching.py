import os
import json
import pandas as pd
import re

# 폴더 경로 설정
image_root = "./processed_data"
label_root = "./label_data"

# 데이터 저장 리스트
data_list = []

for accident_type in os.listdir(image_root):
    image_path = os.path.join(image_root, accident_type)
    label_path = os.path.join(label_root, accident_type.replace("VS_", "VL_"))

    if not os.path.exists(label_path):
        continue

    # frame 번호가 꼬이지 않게 정렬
    video_folder = sorted(os.listdir(image_path), key=lambda x: (x.split("_frame_")[0], int(re.search(r"frame_(\d+)", x).group(1))))

    for video_file in video_folder:
        base_name = video_file.split("_frame_")[0].split(".mp4")[0]
        frame_number = video_file.split("_frame_")[1]  # 프레임 번호 추출
        video_name_with_frame = f"{base_name}_frame_{frame_number}"
        print(video_name_with_frame)

        json_file = os.path.join(label_path, f"{base_name}.json")

        if not os.path.exists(json_file):
            continue

        with open(json_file, "r", encoding="utf-8") as f:
            label_data = json.load(f)

        video_info = label_data["video"]
        rate_a = video_info.get("accident_negligence_rateA", video_info.get("accident_negligence_rate", 0))
        rate_b = video_info.get("accident_negligence_rateB", 100 - rate_a)

        extracted_data = {
            "video_name_with_frame": video_name_with_frame,
            "filming_way": video_info.get("filming_way", "default_filming_way"),
            "video_point_of_view": video_info.get("video_point_of_view", 1),
            "accident_negligence_rateA": video_info.get("accident_negligence_rateA", rate_a),
            "accident_negligence_rateB": video_info.get("accident_negligence_rateB", rate_b),
            "accident_object": video_info.get("accident_object", 0),
            "accident_place": video_info.get("accident_place", 0),
            "accident_place_feature": video_info.get("accident_place_feature", 0),
            "vehicle_a_progress_info": video_info.get("vehicle_a_progress_info", 0),
            "vehicle_b_progress_info": video_info.get("vehicle_b_progress_info", 0),
        }

        # 데이터 저장
        data_list.append(extracted_data)

# DataFrame 생성
df = pd.DataFrame(data_list)

# CSV 파일로 저장 (향후 모델 학습에 활용)
df.to_csv("train_data.csv", index=False, encoding="utf-8")

print("CSV 저장 완료! 🚀")