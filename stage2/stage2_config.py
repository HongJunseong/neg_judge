# config.py

# --- 하이퍼파라미터 ---
FRAMES_PER_SAMPLE = 32    # 샘플당 프레임 개수
BATCH_SIZE        = 16
EPOCHS            = 60
LR                = 1e-4

# 1차: 사고장소 분류 클래스 수
NUM_CLASSES       = 10        # 실제 클래스 수로 변경

# 2차: 멀티태스크 head별 클래스 수
NUM_FEATURES      = 35         # 사고장소 특징
NUM_A_PROGRESS    = 60         # 객체 A 진행 방향
NUM_B_PROGRESS    = 61         # 객체 B 진행 방향