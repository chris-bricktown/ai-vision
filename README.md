# ai-vision

TensorFlow.js와 COCO-SSD(사전 학습 모델)를 이용한 브라우저 웹캠 실시간 객체 인식 웹앱입니다.
순수 클라이언트 사이드 JavaScript로 동작하며, 별도의 서버나 Python 백엔드가 필요 없습니다.

## 데모

GitHub Pages로 배포되어 있습니다: `https://<username>.github.io/ai-vision/`

## 특징

- **서버리스**: 모델 다운로드부터 추론까지 모두 브라우저에서 실행됩니다. 영상은 어디로도 전송되지 않습니다.
- **사전 학습 모델**: [COCO-SSD](https://github.com/tensorflow/tfjs-models/tree/master/coco-ssd)를 CDN에서 불러와 즉시 사용합니다.
- **실시간 인식**: 웹캠 영상 위에 바운딩 박스와 클래스/신뢰도를 실시간으로 오버레이합니다.
- **개/고양이 품종 인식**: COCO-SSD가 "dog"/"cat"을 찾으면 해당 영역을 잘라내 2차로 [MobileNet](https://github.com/tensorflow/tfjs-models/tree/master/mobilenet)(ImageNet 사전 학습)에 넣어 품종 후보를 표시합니다. ImageNet에는 개 품종이 약 120종 포함돼 있어 개는 비교적 정확하지만, 고양이 품종은 4~5종뿐이라 정확도가 낮습니다.
- **한글 인식 결과**: 객체 클래스(80종)와 품종 라벨을 `labels_ko.js`의 매핑 테이블을 통해 한글로 표시합니다. 매핑에 없는 라벨은 영문 그대로 표시됩니다.
- **품종 확정 시 자동 정지 + 대표 이미지**: 품종 분류 신뢰도가 임계값(기본 40%) 이상이면 실시간 인식을 멈추고, 해당 품종의 대표 이미지를 [Wikipedia REST API](https://en.wikipedia.org/api/rest_v1/)에서 가져와 보여줍니다. "다시 인식하기" 버튼으로 재시작할 수 있습니다.
- **카메라 방향에 따른 자동 좌우반전**: 노트북 웹캠처럼 전면 카메라를 쓸 때는 거울처럼 좌우반전되어 보이지만, 휴대폰 후면 카메라(`facingMode: environment`)로 전환되면 실제 세상을 그대로 보여주기 위해 반전을 끕니다.
- **카메라 선택**: 기기에 카메라가 2대 이상 있으면(휴대폰 전/후면, 후면 광각·초광각·망원 등) 드롭다운으로 직접 선택할 수 있습니다. 인식 중에 전환하면 자동으로 재시작됩니다. 카메라 목록은 최초 권한 허용 후에만 라벨과 함께 채워집니다(브라우저 정책). 일부 구형 브라우저(iOS 16 이하 Safari 등)는 후면 카메라를 하나로만 묶어 보여줄 수 있습니다.
  - 일부 안드로이드 기기에서는 Chromium이 같은 물리 카메라를 레거시 Camera1 API(`"camera N, facing back"`)와 Camera2 API(`"camera2 N, facing back"`)로 중복 노출합니다. zoom 등 하드웨어 기능은 Camera2 쪽에서만 동작하므로, 같은 방향에 camera2 항목이 있으면 레거시 중복 항목은 목록에서 자동으로 숨깁니다. (초광각·망원 등 추가 렌즈가 목록에 안 보이는 건 이 중복과는 별개로, 제조사가 해당 렌즈를 Camera2 API에 독립 ID로 노출하지 않아서인 경우가 많습니다 — 이건 브라우저에서 우회할 방법이 없습니다.)
- **줌**: 선택된 카메라가 [MediaTrackCapabilities의 `zoom`](https://developer.mozilla.org/en-US/docs/Web/API/MediaTrackCapabilities/zoom)을 지원하면(주로 Android Chrome) 줌 슬라이더가 나타나 하드웨어/드라이버 수준에서 배율을 조절합니다. CSS나 캔버스로 잘라내는 디지털 줌이 아니라 `track.applyConstraints({ zoom })`으로 카메라 자체에 배율을 요청하는 방식이며, 슬라이더의 최소/최대/단계는 기기가 실제로 보고하는 값을 그대로 읽어오므로 기기별 광학줌 배율 차이(예: 30배 지원 기종)가 자동으로 반영됩니다. 다만 특정 배율 구간이 광학인지 하이브리드(광학+디지털)인지는 OS 카메라 스택이 결정하며 웹 표준에서 구분해서 알려주지 않고, iOS Safari는 이 API 자체를 지원하지 않아 슬라이더가 뜨지 않습니다.

### 품종 분류 모델 교체 (커스텀 학습 모델로)

`breedClassifier.js`는 품종 분류기를 `load()` / `classify()` 인터페이스 뒤에 감싸 두었습니다.
나중에 Oxford-IIIT Pet 등으로 직접 학습한 모델이 생기면:

1. `tensorflowjs_converter`로 TF.js 포맷(model.json + 가중치 샤드)으로 변환해 정적 파일로 호스팅
2. `breedClassifier.js`에 `mobilenetBackend`와 동일한 형태(`load()`, `classify(imageElement, topK)`)의 새 backend 객체를 추가하고 `tf.loadGraphModel`/`tf.loadLayersModel`로 로드, 출력 라벨을 매핑
3. `activeBackend`가 새 backend를 가리키도록 변경

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
├── labels_ko.js                 # 클래스/품종 라벨 한글 매핑
└── .github/workflows/deploy.yml  # GitHub Pages 자동 배포
```
