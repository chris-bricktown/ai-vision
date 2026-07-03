# 커스텀 품종 분류 모델 학습

`../models/custom-breeds/`에 있는 모델을 만든 학습 스크립트입니다. [Oxford-IIIT Pet
Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)의 37개 품종(고양이 12종 + 개 25종) 전체로
학습했고, `../breedClassifier.js`의 `customBackend`가 이걸 사용합니다(기본값은 여전히
MobileNet — 개 품종은 ImageNet이 훨씬 많이 커버하지만, 고양이 품종은 이 커스텀 모델이 12종으로
MobileNet의 5종보다 넓습니다. 바꾸려면 `breedClassifier.js`의 `activeBackend`를 수정). CPU로 8
epoch 학습한 검증 정확도는 약 90.9%입니다.

## 데이터셋

`train.py`의 `CLASSES`가 37개 품종 전체이므로 데이터셋 전체를 내려받아 압축을 풉니다.

```bash
curl -sSL -o images.tar.gz https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
mkdir -p dataset
tar -xzf images.tar.gz -C dataset
```

일부 클래스만 학습하려면 `train.py`의 `CLASSES`를 줄이고, 위 `tar -xzf`에 `--wildcards`로 필요한
클래스 접두어만 추출하면 다운로드/학습 시간을 아낄 수 있습니다(예: `'images/Persian_*'
'images/beagle_*'`).

## 의존성

```bash
pip install tensorflow-cpu tf_keras Pillow
pip install --no-deps tensorflowjs tf_keras tensorflow_decision_forests tensorflow_hub
```

`tensorflowjs`를 의존성째로 설치하면 pip 리졸버가 극도로 느려집니다(tensorflow-cpu와의 버전 제약이
얽혀 10분 넘게 걸리거나 멈춘 것처럼 보일 수 있음). `--no-deps`로 필요한 것만 개별 설치하는 게 훨씬
빠릅니다. `tensorflow_decision_forests`/`jax`는 변환기가 참조는 하지만 실제로 쓰지 않는 경로라서,
설치가 오래 걸리면 빈 스텁 패키지로 대체해도 됩니다(자세한 건 이 프로젝트의 세션 기록 참고).

## 학습 및 변환

```bash
python3 train.py                    # trained_model/saved_model.h5 생성 (CPU로 8 epoch, 37개 클래스 기준 10분 내외)
tensorflowjs_converter --input_format=keras \
  trained_model/saved_model.h5 trained_model/tfjs
cp trained_model/tfjs/* ../models/custom-breeds/
```

## 알아두면 좋은 점

- **`tf.keras`가 아니라 `tf_keras`(Keras 2)를 씁니다.** TensorFlow 2.16+의 `tf.keras`는 기본적으로
  Keras 3인데, Keras 3의 모델 직렬화 포맷(특히 중첩된 Functional 서브모델)이
  `tensorflowjs_converter`의 layers-model 로더와 호환되지 않아 브라우저에서 "Corrupted
  configuration" 같은 에러가 납니다. `tf_keras`로 저장하면 TF.js가 원래 기대하던 Keras 2 포맷으로
  나옵니다.
- **전처리는 모델 그래프 밖에서** 합니다(`train.py`의 `load_image`에서 `(x/127.5) - 1`). Keras
  functional 모델 안에 `preprocess_input`을 레이어로 넣으면 `TrueDivide`/`Subtract` 같은
  op-lambda 레이어로 직렬화되는데 이것도 TF.js가 못 읽습니다. 대신 `breedClassifier.js`의
  `customBackend.classify()`에서 JS로 동일한 정규화를 직접 적용합니다.
- 클래스를 늘리거나 바꾸려면 `train.py`의 `CLASSES`, `breedClassifier.js`의 `CUSTOM_LABELS`,
  `labels_ko.js`의 한글 매핑을 함께 수정해야 합니다.
