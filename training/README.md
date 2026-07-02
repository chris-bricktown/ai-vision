# 커스텀 품종 분류 모델 학습

`../models/custom-breeds/`에 있는 모델을 만든 학습 스크립트입니다. 6개 클래스(고양이 3종 + 개 3종)에
대한 소규모 데모로, `../breedClassifier.js`의 `customBackend`가 이걸 사용할 수 있습니다
(기본값은 여전히 MobileNet — 이 모델로 바꾸려면 `breedClassifier.js`의 `activeBackend`를 수정).

## 데이터셋

[Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)(연구용 라이선스, 37개 품종)에서
`train.py`의 `CLASSES` 목록에 있는 클래스만 사용합니다.

```bash
curl -sSL -o images.tar.gz https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
mkdir -p dataset
tar -xzf images.tar.gz -C dataset --wildcards \
  'images/Persian_*' 'images/Siamese_*' 'images/Bengal_*' \
  'images/beagle_*' 'images/pug_*' 'images/yorkshire_terrier_*'
```

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
python3 train.py                    # trained_model/saved_model.h5 생성 (CPU로 8 epoch, 수 분 소요)
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
