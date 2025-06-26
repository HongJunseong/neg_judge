import os
import cv2
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
import findspark

findspark.init()

# Spark 세션 생성
spark = SparkSession.builder \
    .appName("VideoFrameExtractor") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

# 원본 데이터 폴더 및 저장 폴더 설정
DATA_DIR = "./video_data"  # 사고 유형별 폴더가 있는 폴더
OUTPUT_DIR = "./processed_data"  # 프레임 저장 폴더

# 모든 "VS_" 폴더 탐색
video_folders = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.startswith("TS_") and os.path.isdir(os.path.join(DATA_DIR, f))]

print(f"총 {len(video_folders)}개의 사고 유형 폴더를 처리합니다.")

# 프레임 저장 Schema 설정
frame_schema = StructType([
    StructField("folder_name", StringType(), False),
    StructField("video_name", StringType(), False),
    StructField("frame_index", IntegerType(), False),
    StructField("frame_path", StringType(), False)
])

frame_interval = 3  # 추출할 프레임 간격

# 프레임 추출 함수
def extract_frames():
    frame_data = []

    for folder_path in video_folders:
        folder_name = os.path.basename(folder_path)
        output_folder = os.path.join(OUTPUT_DIR, folder_name)
        os.makedirs(output_folder, exist_ok=True)

        # 해당 폴더 내 동영상 파일 리스트
        video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mov'))]
        print(f"🎬 {folder_name}: {len(video_files)}개의 동영상을 처리합니다.")

        for video_name in video_files:
            video_path = os.path.join(folder_path, video_name)
            video_path = os.path.abspath(video_path)

            print(f"Processing: {video_path}")

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Cannot open video: {video_path}")
                continue

            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    frame_filename = f"{video_name}_frame_{frame_count}.jpg"
                    frame_path = os.path.join(output_folder, frame_filename)

                    # cv2.imencode를 사용하여 한글 경로로 저장
                    frame_type = os.path.splitext(frame_filename)[1]  # 파일 확장자 추출
                    ret, img_arr = cv2.imencode(frame_type, frame, None)

                    if ret:
                        with open(frame_path, mode='w+b') as f:                
                            img_arr.tofile(f)

                    frame_data.append((folder_name, video_name, frame_count, frame_path))

                frame_count += 1

            cap.release()
            print(f"{video_name}: {frame_count // frame_interval} frames extracted")

    return frame_data

# 프레임 추출 실행
frames_data = extract_frames()

# Spark DataFrame 생성
frames_df = spark.createDataFrame(frames_data, schema=frame_schema)

# 결과 확인
frames_df.show(10, truncate=False)

# Spark 종료
spark.stop()
