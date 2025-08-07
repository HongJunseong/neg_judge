from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from neg_judge_pipeline.extract_frame_dag import extract_frames
from neg_judge_pipeline.stage1.stage1_inference import run_stage1_inference
from neg_judge_pipeline.stage2.stage2_inference import run_stage2_inference
from neg_judge_pipeline.stage3.stage3_inference import run_stage3_inference



video_id = "car_accident" # video_id 별로 결과를 나누어 저장

with DAG(
    dag_id='neg_judge_dag',
    start_date=datetime(2025, 8, 1),
    schedule_interval=None,
    catchup=False,
    default_args={
        'owner': 'junseong',
        'retries': 1,
        'retry_delay': timedelta(minutes=1),
    },
) as dag:

    t1 = PythonOperator(
        task_id='extract_frames',
        python_callable=extract_frames,
        op_kwargs={
            'video_path': f'/opt/airflow/data/{video_id}.mp4',       # 원본 영상 데이터
            'output_dir': f'/opt/airflow/data/{video_id}_frames',    # 영상으로부터 추출된 프레임이 저장되는 위치
            'frame_interval': 3,
        },
    )

    t2 = PythonOperator(
        task_id='stage1_inference',
        python_callable=run_stage1_inference,
        op_kwargs={
            'frame_folder': f'/opt/airflow/data/{video_id}_frames',
            'checkpoint_path': '/opt/airflow/models/stage1_best_model.pth',   # 기존 학습된 stage1 best model
            'output_path': f'/opt/airflow/preds/{video_id}/{video_id}_stage1_preds.pth'   # stage1 추론 결과
        },
    )

    t3 = PythonOperator(
        task_id='stage2_inference',
        python_callable=run_stage2_inference,
        op_kwargs={
            'frame_folder': f'/opt/airflow/data/{video_id}_frames',
            'stage1_preds_path': f'/opt/airflow/preds/{video_id}/{video_id}_stage1_preds.pth',   # 기존 학습된 stage1 best model
            'stage2_model_path': '/opt/airflow/models/stage2_best_model.pth',   # 기존 학습된 stage2 best model
            'output_path': f'/opt/airflow/preds/{video_id}/{video_id}_stage2_preds.pth',   # stage2 추론 결과
        },
    )

    t4 = PythonOperator(
        task_id='stage3_inference',
        python_callable=run_stage3_inference,
        op_kwargs={
            'pf_preds_path': f'/opt/airflow/preds/{video_id}/{video_id}_stage1_preds.pth',   #  기존 학습된 stage1 best model
            'stage2_preds_path': f'/opt/airflow/preds/{video_id}/{video_id}_stage2_preds.pth',   # 기존 학습된 stage2 best model
            'stage3_model_path': '/opt/airflow/models/stage3_best_model.pth'   # 기존 학습된 stage3 best model
        },
    )

    t1 >> t2 >> t3 >> t4