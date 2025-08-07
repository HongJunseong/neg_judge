FROM apache/airflow:2.7.3-python3.11

USER root

COPY requirements.txt /requirements.txt
RUN python -m pip install --upgrade pip
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0


USER airflow

RUN pip install pytorchvideo
RUN pip install scikit-learn
RUN pip install numpy==2.0.2 pandas==2.2.3
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install apache-airflow==${AIRFLOW_VERSION} -r /requirements.txt
