FROM apache/airflow:2.5.1-python3.8

USER root

COPY requirements.txt /requirements.txt
RUN python -m pip install --upgrade pip
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0


USER airflow

RUN pip install pytorchvideo
RUN pip install scikit-learn pandas
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install apache-airflow==${AIRFLOW_VERSION} -r /requirements.txt
