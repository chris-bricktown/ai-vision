# ai-vision

TensorFlow.js와 COCO-SSD(사전 학습 모델)를 이용한 브라우저 웹캠 실시간 객체 인식 웹앱입니다.
순수 클라이언트 사이드 JavaScript로 동작하며, 별도의 서버나 Python 백엔드가 필요 없습니다.

## 데모

GitHub Pages로 배포되어 있습니다: `https://<username>.github.io/ai-vision/`

## 특징

- **서버리스**: 모델 다운로드부터 추론까지 모두 브라우저에서 실행됩니다. 영상은 어디로도 전송되지 않습니다.
- **사전 학습 모델**: [COCO-SSD](https://github.com/tensorflow/tfjs-models/tree/master/coco-ssd)를 CDN에서 불러와 즉시 사용합니다.
- **실시간 인식**: 웹캠 영상 위에 바운딩 박스와 클래스/신뢰도를 실시간으로 오버레이합니다.

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
└── .github/workflows/deploy.yml  # GitHub Pages 자동 배포
```
