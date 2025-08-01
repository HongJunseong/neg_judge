# 딥러닝 기반 교통사고 과실 비율 판단 시스템

<br>

## 프로젝트 요약

교통사고 발생 시 블랙박스 영상을 활용해 과실 비율을 자동으로 산정하는 시스템은 보험 심사와 분쟁 조정을 빠르고 객관적으로 수행하는 데 큰 도움이 될 수 있습니다.

본 프로젝트에서는 사고 영상의 프레임 시퀀스를 입력으로 받아, 1단계에서는 ResNet34 기반의 TSN 모델을 통해 사고 발생 장소를 분류하고, 2단계에서는 해당 예측 결과와 영상 피처를 결합하여 SlowFast 백본을 기반으로 사고 특징 및 차량 A·B의 진행 방향을 동시에 예측합니다.
마지막으로, 앞 단계의 예측값들을 결합한 벡터를 MLP 분류기에 입력해 과실 비율을 구간별로 판단하였습니다. 각 단계에서는 Optuna를 활용한 하이퍼파라미터 최적화를 통해 학습의 안정성과 성능을 높였습니다.

## 프로젝트 배경

최근 자동차 보급률이 급격히 증가하면서 교통사고 발생 건수와 그에 따른 피해 규모도 꾸준히 늘고 있습니다.
사고가 발생하면 당사자 간 과실 비율을 산정하여 보험 청구 및 법적 분쟁을 신속하게 처리해야 하지만,
기존의 방식(현장 조사관의 육안 판독, 블랙박스 영상의 수기 분석, 전문가 감정 등)은 주관성이 개입되며 시간이 오래 걸리는 한계가 있습니다.

특히 동일한 블랙박스 영상을 두고도 조사자나 전문가에 따라 해석이 달라지는 경우가 많아,
객관적인 판단이 어려워지고, 이로 인해 보험 처리 및 분쟁 조정 과정에서 불필요한 논쟁이 발생하기도 합니다.

한편, 최근 딥러닝 기반 컴퓨터 비전 기술의 발전으로 인해
블랙박스 영상으로부터 사고 장소, 차량 궤적, 충돌 상황 등 다양한 정보를 정확하게 추출할 수 있는 가능성이 열리고 있습니다.
멀티프레임 기반의 분석 기법은 사고 전후의 맥락을 고려할 수 있으며,
멀티태스크 학습을 통해 사고 장소, 사고 특징, 차량 진행 방향을 동시에 예측함으로써
더 정밀하고 일관된 과실 비율 산정이 가능합니다.

이에 따라, 블랙박스 영상만으로도 과실 비율을 객관적이고 신속하게 판단할 수 있는 자동화 시스템의 필요성이 점차 커지고 있습니다.


## 기존 접근 방식의 한계 및 해결 방향

### 1. **사고 장소 클래스의 한정성 문제**
   
기존 연구에서는 사고 장소를 교차로, 횡단보도 등 일부 대표 구역으로만 한정하여 학습하는 경우가 많았습니다.
그러나 실제 사고는 고속도로, 복합 교차로, 좁은 골목길 등 훨씬 다양한 장소에서 발생하며,
이처럼 제한된 클래스만 학습한 모델은 학습에 포함되지 않은 장소가 입력될 경우 오분류할 가능성이 높습니다.

> **개선 방향** <br>
> 본 프로젝트는 보다 다양한 사고 장소를 포함한 데이터셋을 구성하여,
> 현실의 다양한 도로 환경을 반영할 수 있도록 설계하였습니다.
> 이를 통해 장소의 다양성을 확보하고, 새로운 환경에서도 안정적으로 대응할 수 있는 모델을 구축하고자 하였습니다.


### 2. **TSN 기반 모델의 모션 정보 손실**

기존의 행동 인식 모델로 자주 사용된 TSN (Temporal Segment Network) 은
여러 시점의 대표 프레임만을 2D-CNN으로 처리하고 평균화하는 방식으로 연산 효율은 높지만,
짧은 시간 내 발생하는 급격한 움직임이나 충돌 직전의 미세한 변화를 놓칠 수 있는 단점이 있습니다.
예를 들어, 측면 충돌, 급가속/급정지 등은 대표 프레임 평균화 과정에서 희석될 가능성이 있습니다.

> **개선 방향** <br>
> 이 문제를 보완하기 위해 본 프로젝트에서는 SlowFast 네트워크를 도입했습니다.
> SlowFast는 느린 경로(Slow Pathway) 와 빠른 경로(Fast Pathway) 를 병렬로 구성하여,
> 장기적인 맥락과 순간적인 모션 정보를 모두 정밀하게 포착할 수 있습니다.
> 이를 통해 사고 특징과 차량 진행 방향에 대한 예측 정확도를 TSN 대비 향상시킬 수 있었습니다.


### 3. **수작업 기반 입력 방식의 불편함**
   
기존 연구에서는 사고 과실 비율 예측 시,
사고 장소, 사고 특징, 차량 진행 방향 등의 정보를 사용자가 수작업으로 입력하거나 외부 시스템의 결과를 추가 입력해야 했습니다.
이는 사용자의 부담을 증가시키고, 실시간 처리에 제약을 줄 수 있습니다.

> **개선 방향** <br>
> 본 프로젝트는 블랙박스 영상만을 입력으로 받아,
> Stage 1과 Stage 2에서 사고 장소·사고 특징·차량 진행 방향을 자동으로 추론하고,
> 해당 로짓을 그대로 Stage 3의 MLP 분류기에 전달하여 과실 비율을 구간별로 자동 예측합니다.
> End-to-End 구조를 통해, 별도 수작업 없이도 실무 현장에 바로 적용 가능한 자동화 시스템을 구현하고자 하였습니다.

## 데이터 구성

본 프로젝트는 AI Hub에서 제공하는 **교통사고 블랙박스 영상 데이터셋**을 기반으로 구성되었습니다.  
총 **4223개의 사고 영상**을 수집하여 사용하였으며, 각 영상은 사고 발생 전후의 상황을 담고 있습니다.

### 1. 입력 데이터
- 각 영상으로부터 추출된 **프레임 시퀀스**를 모델 입력으로 사용
- 한 샘플은 여러 장의 프레임 이미지로 구성된 시퀀스입니다

### 2. 주요 라벨
- 각 사고 영상은 다음과 같은 정보를 기반으로 학습에 사용됩니다:
  - 사고 발생 장소
  - 사고 특징
  - 사고 객체 A와 B의 진행 방향
  - 최종 과실 비율 구간 (예: 30:70, 50:50 등)

### 3. 전처리 및 구성 방식
- 영상에서 일정 간격으로 프레임을 추출하여 시퀀스 데이터 생성
- **극소수 클래스(샘플 수가 매우 적은 라벨)는 제거**하여 학습 안정성 확보
- 데이터는 학습/검증/테스트 세트로 **8:1:1 비율로 분할**하며, 라벨 분포를 고려해 균형 있게 구성


## 전체 시스템 구성 요약

![image](https://github.com/user-attachments/assets/68ec87c7-790f-47f1-8257-b3ec9774d4df)


### Stage 1: 사고 장소 분류 (Scene Classification)
- 입력: 사고 영상에서 추출한 프레임 시퀀스
- 모델: TSN (Temporal Segment Network) + ResNet34 백본
- 출력: 사고 발생 장소 분류 결과 (다중 클래스)
- 역할: 사고 유형 판단의 기반이 되는 장소 정보를 사전 예측하여 후속 모델에 활용


### Stage 2: 사고 특징 및 차량 진행 방향 예측 (Multi-task Learning)
- 입력:
  - 프레임 시퀀스
  - Stage 1에서 예측된 사고 장소 결과
- 모델: SlowFast 네트워크 기반의 멀티태스크 분류 모델
- 출력:
  - 사고 특징
  - 차량 A의 진행 방향
  - 차량 B의 진행 방향
- 특징:
  - **느린 경로와 빠른 경로**를 함께 사용하는 구조로, 장기 맥락과 짧은 순간의 모션을 동시에 인식
  - 세 가지 속성을 동시에 예측하여 효율성과 정밀도 확보


### Stage 3: 과실 비율 구간 분류 (Negligence Ratio Classification)
- 입력: Stage 1과 Stage 2의 예측 결과 벡터를 통합
- 모델: MLP (다층 퍼셉트론) 기반 분류기
- 출력: 과실 비율 구간 클래스 (예: 0:100, 10:90, ..., 100:0)
- 특징:
  - End-to-End 파이프라인의 최종 출력 단계
 
> 본 시스템은 각 단계를 개별적으로 학습하고 연결하는 방식으로 구성되어 있으며,  
> 각 단계의 예측 결과는 후속 단계의 입력으로 활용됩니다.  
> 이러한 구조는 사고 관련 정보를 **단계적으로 분리하여 예측**함으로써,  
> 복잡한 판단 과정을 모듈화하고, 영상만으로도 **자동·객관적으로 과실 비율을 산정할 수 있는 구조**를 제공합니다.


## 모델 성능 및 결과

### Stage 1: (사고 장소) 모델 분류 결과

Stage1 best model train, val - Accuracy, Loss result :

|Dataset|	Loss (CrossEntorpy)	|Top-1 Accuracy|
|:------:|:--------:|:----------:|
|Train|	0.529|	0.996|
|Validation|	1.300	|	0.715	|

Stage1 train, val - Accuracy , Loss curve :

![image](https://github.com/user-attachments/assets/8388c2cb-c3a4-470c-841d-1fede6eb71b7)


Stage1 best model test - Top-1, Top-3 Accuracy :

|Dataset|	Top-1 Accuracy	|Top-3 Accuracy|
|:------:|:--------:|:----------:|
|Test|	0.700|	0.899|

<br>

### Stage2 (사고 특징, 객체 A 진행 방향, 객체 B 진행 방향) 모델 분류 결과

Stage2 best model train, val - Loss result :

|Dataset|	Loss (Focal Loss)	|
|:------:|:--------:|
|Train|	0.064|
|Validation|	1.300	|

Stage2 best model train, val - Top-1 Accuracy result :

|Dataset|	Feature Top-1 Accuracy|	Object A Top-1 Accuracy|	Object B Top-1 Accuracy|
|:------:|:--------:|:----------:|:----------:|
|Train	|0.982|	0.959|	0.963|
|Validation|	0.596|	0.500|	0.490|

Stage2 train, val - Accuracy , Loss curve :

![image](https://github.com/user-attachments/assets/4982138a-38a9-4ce0-a757-2f59e2fdb875)

Stage2 best model test - Top-1 Accuracy :
|Dataset|	Feature Top-1 Accuracy|	Object A Top-1 Accuracy|	Object B Top-1 Accuracy|
|:------:|:--------:|:----------:|:----------:|
|Test	|0.612|	0.490|	0.472|

Stage2 best model test - Top-3 Accuracy :
|Dataset|	Feature Top-3 Accuracy|	Object A Top-3 Accuracy|	Object B Top-3 Accuracy|
|:------:|:--------:|:----------:|:----------:|
|Test	|0.788|	0.653|	0.648|

<br>

### Stage3 (최종 과실 비율) 모델 분류 결과

Stage3 best model train, val - Loss result :
|Dataset|	Loss (CrossEntorpy)	|Top-1 Accuracy|
|:------:|:--------:|:----------:|
|Train|	0.873|	0.878|
|Validation|	1.758	|	0.557	|

Stage3 train, val - Accuracy , Loss curve :

![image](https://github.com/user-attachments/assets/6aec2fa4-2ef9-45d1-b84a-2e53bd315ea9)

Stage3 best model test - Top-1 Accuracy :
|Dataset|Top-1 Accuracy|
|:------:|:--------:|
|Test|	0.510|

Stage3 model ±1 class Accuracy :
|Dataset|1-Off  Accuracy|
|:------:|:--------:|
|Test|	0.663|

## 기술적 도전과제

### 1. 로컬 환경 한계를 극복한 고성능 GPU 서버 활용
프로젝트 초기에는 개인 PC 환경에서 모델을 학습했으나,
GPU 메모리 부족과 느린 연산 속도로 인해 복잡한 모델 구조나 고부하 연산을 실행하는 데 어려움이 있었습니다.
이러한 제약을 극복하기 위해 고성능 GPU 서버를 도입하여,
보다 큰 규모의 학습, 고해상도 입력 처리, 병렬 기반의 하이퍼파라미터 탐색 등
복잡한 실험을 원활하게 수행할 수 있는 환경을 구축하였습니다.

그 결과, 연산 효율성과 실험 범위가 크게 향상되었으며,
로컬 환경에서는 한계가 있었던 다양한 시도들을 안정적으로 실행할 수 있게 되었습니다.

### 2. YOLO/DeepSORT 기반 객체 추적의 현실적 한계
사고 영상에는 사고 대상 외에도 다수의 차량, 자전거, 보행자 등이 함께 등장하기 때문에,
YOLO 및 DeepSORT를 적용하더라도 충돌에 실제로 관여한 객체만을 정확히 추적하기 어렵다는 한계가 있었습니다.
또한, 프레임 간 위치 차이를 기반으로 상대 속도를 추정하려 했으나,
정확도가 낮고 연산 시간도 과도하게 증가하여 모델에 적용이 어려웠습니다.

이러한 기술적 제약으로 인해 객체 검출·추적 모듈은 제외하고,
대신 SlowFast 모델을 통해 사고 객체의 모션 정보를 직접 학습에 반영하는 방식으로 전환하였습니다.

### 3. 다단계 구조에서의 오차 누적과 보정 시도
본 시스템은 Stage 1 → Stage 2 → Stage 3 순으로 예측 결과를 순차적으로 전달하는 구조이기 때문에,
초기 단계(Stage 1)의 예측 오류가 후속 단계 성능에 영향을 미치는 구조적 한계가 존재합니다.

이를 완화하기 위해, Stage 1의 예측 결과를 hardmax가 아닌 softmax 확률 분포 형태로 전달하고,
temperature 조정 기법을 통해 불확실성이 큰 예측값에 대해 보다 완만한 확률 분포를 생성함으로써
Stage 2에서 더 안정적인 학습이 가능하도록 보정하였습니다.

이러한 방식은 예측 오류가 다음 단계에 과도하게 반영되는 것을 줄이고,
다단계 구조의 오차 누적 문제를 정량적으로 완화하기 위한 접근으로 작용하였습니다.


## 인프라 및 개발 환경

### 하드웨어
|GPU	|CPU	|Memory|
|:----:|:------:|:------:|
|NVIDIA RTX 3090 (24GB)	|Intel(R) Xeon(R) E-2334 @ 3.40GHz	|32GB RAM|

### 소프트웨어
|Python	|PyTorch |CUDA	| Spark |
|:----:|:------:|:------:|:------:|
|3.11.4 |	2.5.1 | 11.8	|3.5.5 |
