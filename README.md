# 2023 Fake or Real : 생성 이미지 판별 경진대회
## [이미지] 생성 이미지 판별

### 코드 구조

```
${PROJECT}
├── config/
│   ├── train_config.yaml
│   └── predict_config.yaml
├── ensemble/
│   ├── ensemble.py
│   └── ensemble_config.yaml
├── models/
│   ├── CLIPModel.py
│   ├── TFModel.py
│   ├── effnet_google.py
│   ├── effnet.py
│   └── utils.py
├── modules/
│   ├── datasets.py
│   ├── earlystoppers.py
│   ├── losses.py
│   ├── metrics.py
│   ├── optimizers.py
│   ├── recorders.py
│   ├── trainer.py
│   └── utils.py
├── README.md
├── train.py
└── predict.py
```

- config: 학습/추론에 필요한 하이퍼파라미터 등을 기록하는 yaml 파일
- ensemble
    - ensemble.py: 앙상블 클래스
    - ensemble_config.yaml: 앙상블의 선택할 csv파일을 기록하는 yaml 파일
- models
    - CLIPModel.py: CLIP 기반 모델 클래스
    - TFModel.py: Transformer 기반 모델 클래스
    - effnet_google.py: Timm을 사용하지 않고 Google기반의의 Efficinetnet 모델 클래스
    - effnet.py: Efficinetnet 모델 클래스
    - utils.py: config에서 지정한 모델 클래스를 불러와 리턴하는 파일
- modules
    - datasets.py: dataset 클래스
    - earlystoppers.py: loss가 지정된 에폭 수 이상 개선되지 않을 경우 학습을 멈추는 early stopper 클래스
    - losses.py: config에서 지정한 loss function을 리턴
    - metrics.py: config에서 지정한 metric을 리턴
    - optimizers.py: config에서 지정한 optimizer를 리턴
    - recorders.py: 로그와 learnig curve 등을 기록
    - trainer.py: 에폭 별로 수행할 학습 과정
    - utils.py: 여러 확장자 파일을 불러오거나 여러 확장자로 저장하는 등의 함수가 포함된 파일
- train.py: 학습 시 실행하는 코드
- predict.py: 추론 시 실행하는 코드


---

### 학습 process

1. 데이터 폴더 준비
    1. 아래 구조와 같이 데이터 폴더를 생성
```
${DATA}
├── train/
│   ├── real_images/
│   │   ├── 'real_00000.png'
│   │   ├── 'real_00001.png'
│   │   ├── 'real_00002.png'
│   │   ├── 'real_00003.png'
│   │   └──  ...
│   └── fake_images/
│       ├── 'fake_00000.png'
│       ├── 'fake_00001.png'
│       ├── 'fake_00002.png'
│       ├── 'fake_00003.png'
│       └──  ...
└── test/
    └── images/
        ├── 'test_00000.png'
        ├── 'test_00000.png'
        ├── 'test_00000.png'
        ├── 'test_00000.png'
        └──  ...
```

2. 'config/train_config.yaml' 수정
    1. DIRECTORY/dataset: '{DATA}/train'로 경로 지정
    2. 이외 파라미터 수정
3. 'python train.py' 실행
4. 'results/train/'내에 결과 (모델 가중치, 학습 log 등)가 저장됨


### 추론 Process

1. 'config/predict_config.yaml' 수정
    1. DIRECTORY/dataset: '{DATA}/test/images'로 경로 지정
    2. DIRECTORY/sample_submission_path: sample_subission.csv 파일의 경로 지정
    3. TRAIN/train_serial: 학습된 모델 가중치 및 하이퍼파라미터를 불러올 train serial number (result/train 내 폴더명) 지정
2. 'python predict.py' 실행
3. 'results/predict/' 내에 결과 파일(predictions.csv)이 저장됨

---

### 학습 진행 방법 


모델의 최종 출력단 last hidden state에서 CLS를 출력하여 sigmoid를 하는 방식 (BCEWITHLOGITSLOSS를 사용)으로 학습을 진행

### 추가 데이터셋 


Hugging Face 의 AiorNot

https://huggingface.co/spaces/competitions/aiornot 

### 앙상블


Hard Voting기반 앙상블 사용용
     
