# ai-vision

TensorFlow.js와 COCO-SSD(사전 학습 모델)를 이용한 브라우저 웹캠 실시간 객체 인식 웹앱입니다.
순수 클라이언트 사이드 JavaScript로 동작하며, 별도의 서버나 Python 백엔드가 필요 없습니다.

## 데모

GitHub Pages로 배포되어 있습니다: `https://<username>.github.io/ai-vision/`

## 특징

- **서버리스**: 모델 다운로드부터 추론까지 모두 브라우저에서 실행됩니다. 영상은 어디로도 전송되지 않습니다.
- **사전 학습 모델**: [COCO-SSD](https://github.com/tensorflow/tfjs-models/tree/master/coco-ssd)를 CDN에서 불러와 즉시 사용합니다.
- **실시간 인식**: 웹캠 영상 위에 바운딩 박스와 클래스/신뢰도를 실시간으로 오버레이합니다.
- **개/고양이 품종 인식**: COCO-SSD가 "dog"/"cat"을 찾으면 해당 영역을 잘라내 2차로 품종 분류기에 넣습니다. 기본 동작은 [MobileNet](https://github.com/tensorflow/tfjs-models/tree/master/mobilenet)(ImageNet 사전 학습, 개 품종 약 120종·고양이 품종 5종)과 자체 학습한 커스텀 모델(Oxford-IIIT Pet Dataset 37개 품종 전체, 고양이 12종·개 25종)을 **둘 다 매번 실행해서 신뢰도가 더 높은 쪽의 결과를 보여주는** 방식입니다 — MobileNet은 개 품종은 넓게 커버하지만 고양이 품종이 좁고, 커스텀 모델은 반대라 서로의 약점을 보완합니다. MobileNet은 품종 전용 모델이 아니라 ImageNet 1000개 클래스 전체를 분류하는 범용 모델이라, COCO-SSD가 "고양이"라고 정확히 잡아낸 영역에도 전혀 관계없는 클래스(예: "ice bear")가 1순위로 나올 수 있습니다 — 이를 방지하기 위해 전체 1000개 클래스 확률을 받아온 뒤 COCO-SSD가 판단한 카테고리(dog/cat)에 해당하는 known 품종 목록(`labels_ko.js`의 `DOG_BREED_LABELS_KO`/`CAT_BREED_LABELS_KO`)에 있는 것만 후보로 남기고, 그중 확률이 가장 높은 걸 보여줍니다.
- **한글 인식 결과**: 객체 클래스(80종)와 품종 라벨을 `labels_ko.js`의 매핑 테이블을 통해 한글로 표시합니다. 매핑에 없는 라벨은 영문 그대로 표시됩니다.
- **품종 확정 시 자동 정지 + 대표 이미지**: 품종 분류 신뢰도가 임계값(기본 40%) 이상이면 실시간 인식을 멈추고, 해당 품종의 대표 이미지를 [Wikipedia REST API](https://en.wikipedia.org/api/rest_v1/)에서 가져와 보여줍니다. "다시 인식하기" 버튼으로 재시작할 수 있습니다.
- **카메라 방향에 따른 자동 좌우반전**: 노트북 웹캠처럼 전면 카메라를 쓸 때는 거울처럼 좌우반전되어 보이지만, 휴대폰 후면 카메라(`facingMode: environment`)로 전환되면 실제 세상을 그대로 보여주기 위해 반전을 끕니다.
- **카메라 선택**: 기기에 카메라가 2대 이상 있으면(휴대폰 전/후면, 후면 광각·초광각·망원 등) 드롭다운으로 직접 선택할 수 있습니다. 인식 중에 전환하면 자동으로 재시작됩니다. 카메라 목록은 최초 권한 허용 후에만 라벨과 함께 채워집니다(브라우저 정책). 일부 구형 브라우저(iOS 16 이하 Safari 등)는 후면 카메라를 하나로만 묶어 보여줄 수 있습니다.
  - 일부 안드로이드 기기에서는 Chromium이 같은 물리 카메라를 레거시 Camera1 API(`"camera N, facing back"`)와 Camera2 API(`"camera2 N, facing back"`)로 중복 노출합니다. zoom 등 하드웨어 기능은 Camera2 쪽에서만 동작하므로, 같은 방향에 camera2 항목이 있으면 레거시 중복 항목은 목록에서 자동으로 숨깁니다. (초광각·망원 등 추가 렌즈가 목록에 안 보이는 건 이 중복과는 별개로, 제조사가 해당 렌즈를 Camera2 API에 독립 ID로 노출하지 않아서인 경우가 많습니다 — 이건 브라우저에서 우회할 방법이 없습니다.)
- **줌**: 선택된 카메라가 [MediaTrackCapabilities의 `zoom`](https://developer.mozilla.org/en-US/docs/Web/API/MediaTrackCapabilities/zoom)을 지원하면(주로 Android Chrome) 줌 슬라이더가 나타나 하드웨어/드라이버 수준에서 배율을 조절합니다. CSS나 캔버스로 잘라내는 디지털 줌이 아니라 `track.applyConstraints({ zoom })`으로 카메라 자체에 배율을 요청하는 방식이며, 슬라이더의 최소/최대/단계는 기기가 실제로 보고하는 값을 그대로 읽어오므로 기기별 광학줌 배율 차이(예: 30배 지원 기종)가 자동으로 반영됩니다. 다만 특정 배율 구간이 광학인지 하이브리드(광학+디지털)인지는 OS 카메라 스택이 결정하며 웹 표준에서 구분해서 알려주지 않고, iOS Safari는 이 API 자체를 지원하지 않아 슬라이더가 뜨지 않습니다.
- **나만의 사물 학습하기**: "+ 나만의 사물 학습하기" 패널에서 이름(타이틀)을 정하고 웹캠 촬영 또는 이미지 업로드로 사진을 몇 장만 등록하면 그 자리에서 바로 인식할 수 있습니다. 서버나 별도 학습 과정 없이, MobileNet으로 뽑은 이미지 임베딩에 [k-NN 분류기](https://github.com/tensorflow/tfjs-models/tree/master/knn-classifier)를 붙이는 방식(Teachable Machine과 같은 전이학습 기법)이라 사진을 추가하는 즉시 그게 학습입니다. **클래스가 최소 2개 있어야** "내 모델로도 인식하기"를 켤 수 있습니다 — 클래스가 1개뿐이면 k-NN이 비교할 대상이 없어 무엇을 비추든 항상 그 클래스로 인식되므로, 인식 대상 외에 "배경" 같은 클래스도 하나 추가하는 걸 권장합니다. 판정 결과는 실시간으로 표시되며 그때그때 가장 비슷했던 학습 사진과 신뢰도·유사도를 함께 보여줍니다. 가장 비슷한 후보의 유사도가 70%를 넘으면 자동으로 하나를 정하는 대신 스캔을 멈추고, 상위 후보(최대 3개, 각각 학습 사진·신뢰도·유사도 표시)를 보여줘서 **직접 어떤 것인지 선택**하게 합니다. 고른 사진은 그 자리에서 해당 클래스의 학습 데이터로 추가되어 다음부터 더 정확하게 인식합니다(모두 아니면 "모두 아님 · 다시 스캔"으로 넘길 수 있습니다). 후보 목록의 각 항목은 항상 자기 자신의 학습 사진 기준으로 신뢰도·유사도가 계산되므로, 이름과 매칭 사진이 서로 다른 클래스를 가리키는 일은 없습니다. "모델 내보내기/불러오기"로 등록한 클래스를 JSON 파일로 저장했다가 다시 불러올 수 있습니다(단, 원본 사진은 저장하지 않아 불러온 클래스는 매칭 사진 미리보기 없이 라벨/신뢰도만 표시됩니다). COCO-SSD/품종 분류 파이프라인과는 완전히 독립적으로 동작합니다(`customTrainer.js`, `trainerUI.js`).

### 품종 분류 모델 교체 (커스텀 학습 모델로)

`breedClassifier.js`는 품종 분류기를 `load()` / `classify()` 인터페이스 뒤에 감싸 두었습니다.
`models/custom-breeds/`에는 Oxford-IIIT Pet Dataset 37개 품종(고양이 12종 + 개 25종) 전체로 직접
학습한 모델이 들어 있고, `breedClassifier.js`의 `customBackend`로 구현돼 있습니다(학습 스크립트와
과정은 [`training/README.md`](training/README.md) 참고, 검증 정확도 약 90.9%).

기본 `activeBackend`는 `mobilenetBackend`도 `customBackend`도 아니라 `combinedBackend`로, 매 크롭마다
둘 다 실행해서(`Promise.all`) 신뢰도가 더 높은 쪽 결과를 그대로 씁니다. 두 모델의 확률값이 완벽하게
같은 척도는 아니지만(클래스 수가 적을수록 softmax 확률이 더 뾰족하게 나오는 경향), 서로 독립적인
두 모델의 의견을 결합하는 단순하고 실용적인 방법으로 채택했습니다.

다른 모델로 바꾸거나 추가하려면:

1. `tensorflowjs_converter`로 TF.js 포맷(model.json + 가중치 샤드)으로 변환해 정적 파일로 호스팅
2. `breedClassifier.js`에 `mobilenetBackend`/`customBackend`와 동일한 형태(`load()`, `classify(imageElement, topK, cocoClass)`)의 새 backend 객체를 추가하고 `tf.loadGraphModel`/`tf.loadLayersModel`로 로드, 출력 라벨을 매핑
3. `activeBackend`가 새 backend를 가리키도록 변경하거나, `combinedBackend`에 합류시켜 함께 비교하도록 수정

`app.js` 쪽은 수정할 필요가 없습니다.

## 로컬 실행

빌드 과정이 없는 정적 사이트이므로 정적 파일 서버로 바로 실행할 수 있습니다.

```bash
npx serve .
# 또는
python3 -m http.server 8000
```

브라우저에서 열고 "웹캠 시작" 버튼을 눌러 카메라 접근을 허용하세요.

> 카메라 접근은 보안상 `https://` 또는 `http://localhost` 환경에서만 동작합니다.

## 배포

`main` 브랜치에 푸시하면 `.github/workflows/deploy.yml` 워크플로가 자동으로 GitHub Pages에 배포합니다.
저장소 설정의 **Settings > Pages > Source**에서 "GitHub Actions"를 선택해야 합니다.

## 파일 구조

```
.
├── index.html               # 메인 페이지
├── style.css                 # 스타일
├── app.js                     # 웹캠 접근, 모델 로드, 추론 루프
├── breedClassifier.js          # 품종 분류기 (교체 가능한 backend 인터페이스)
├── customTrainer.js             # 브라우저 내 즉석 학습(k-NN + MobileNet 임베딩)
├── trainerUI.js                  # "나만의 사물 학습하기" 패널 UI 연결
├── labels_ko.js                 # 클래스/품종 라벨 한글 매핑
├── models/custom-breeds/         # 커스텀 학습 모델 예시 (TF.js layers-model)
├── training/                      # 위 모델을 만든 학습 스크립트 (train.py, README.md)
└── .github/workflows/deploy.yml  # GitHub Pages 자동 배포
```
