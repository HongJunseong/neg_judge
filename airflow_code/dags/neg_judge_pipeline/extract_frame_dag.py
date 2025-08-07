import os
import cv2


def extract_frames(video_path: str, output_dir: str, frame_interval: int = 3):
    os.makedirs(output_dir, exist_ok=True)

    video_name = os.path.basename(video_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"[ERROR] Cannot open video file at: {video_path}. "
                                f"Check if the file exists inside the container.")

    count = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            frame_filename = f"{os.path.splitext(video_name)[0]}_frame_{count}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)

            success, encoded = cv2.imencode(".jpg", frame)
            if success:
                with open(frame_path, "wb") as f:
                    f.write(encoded)
                saved += 1
        count += 1

    cap.release()
    print(f"[extract_frames] {saved} frames saved to: {output_dir}")