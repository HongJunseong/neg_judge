import os
import shutil
import random

def split_dataset_by_video(source_dir, train_dir, val_dir, test_dir, val_ratio=0.15, test_ratio=0.15):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    class_folders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]

    for class_folder in class_folders:
        src_path = os.path.join(source_dir, class_folder)
        train_path = os.path.join(train_dir, class_folder)
        val_path = os.path.join(val_dir, class_folder)
        test_path = os.path.join(test_dir, class_folder)
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(val_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)

        images = [f for f in os.listdir(src_path) if f.endswith(('.jpg', '.png'))]

        # 그룹을 video_id 단위로 묶기
        video_groups = {}
        for img in images:
            video_id = img.split("_frame_")[0]
            video_groups.setdefault(video_id, []).append(img)

        video_ids = list(video_groups.keys())
        random.shuffle(video_ids)

        total_videos = len(video_ids)
        val_count = int(total_videos * val_ratio)
        test_count = int(total_videos * test_ratio)
        train_count = total_videos - val_count - test_count

        train_ids = video_ids[:train_count]
        val_ids = video_ids[train_count:train_count+val_count]
        test_ids = video_ids[train_count+val_count:]

        for video_id in train_ids:
            for img in video_groups[video_id]:
                shutil.copy(os.path.join(src_path, img), os.path.join(train_path, img))

        for video_id in val_ids:
            for img in video_groups[video_id]:
                shutil.copy(os.path.join(src_path, img), os.path.join(val_path, img))

        for video_id in test_ids:
            for img in video_groups[video_id]:
                shutil.copy(os.path.join(src_path, img), os.path.join(test_path, img))

        print(f"{class_folder}: {len(train_ids)} videos to train / {len(val_ids)} to val / {len(test_ids)} to test")


split_dataset_by_video("./processed_data", "data/train", "data/val", "data/test", val_ratio=0.15, test_ratio=0.15)
