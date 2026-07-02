(() => {
  const video = document.getElementById("webcam");
  const canvas = document.getElementById("overlay");
  const ctx = canvas.getContext("2d");
  const stage = document.getElementById("stage");
  const startBtn = document.getElementById("start-btn");
  const stopBtn = document.getElementById("stop-btn");
  const statusEl = document.getElementById("status");
  const fpsEl = document.getElementById("fps");
  const detectionsEl = document.getElementById("detections");
  const resultPanel = document.getElementById("result-panel");
  const resultTitle = document.getElementById("result-title");
  const resultImage = document.getElementById("result-image");
  const resultCaption = document.getElementById("result-caption");
  const rescanBtn = document.getElementById("rescan-btn");
  const cameraSelect = document.getElementById("camera-select");

  const DETECTION_INTERVAL_MS = 100;
  const BREED_INTERVAL_MS = 700;
  const MIN_SCORE = 0.5;
  const BREED_CONFIRM_THRESHOLD = 0.4;
  const BOX_COLOR = "#37e07a";

  const cropCanvas = document.createElement("canvas");
  const cropCtx = cropCanvas.getContext("2d");

  let mirrored = true;
  let selectedDeviceId = null;
  let model = null;
  let breedReady = false;
  let lastBreedResults = [];
  let stream = null;
  let running = false;
  let rafId = null;
  let lastDetectTime = 0;
  let lastBreedTime = 0;
  let lastFrameTime = performance.now();

  function setStatus(text) {
    statusEl.textContent = text;
  }

  async function ensureModel() {
    if (model) return model;
    if (typeof tf === "undefined" || typeof cocoSsd === "undefined") {
      throw new Error("모델 라이브러리를 불러오지 못했습니다. 네트워크 연결을 확인하세요.");
    }
    setStatus("모델을 불러오는 중...");
    model = await cocoSsd.load();
    return model;
  }

  async function ensureBreedModel() {
    try {
      await BreedClassifier.load();
      breedReady = true;
    } catch (err) {
      console.warn("품종 분류 모델을 불러오지 못했습니다:", err.message);
      breedReady = false;
    }
  }

  function getVideoConstraints() {
    return selectedDeviceId ? { deviceId: { exact: selectedDeviceId } } : { facingMode: "environment" };
  }

  async function populateCameraList() {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const cameras = devices.filter((d) => d.kind === "videoinput");

      cameraSelect.innerHTML = "";
      cameras.forEach((cam, i) => {
        const option = document.createElement("option");
        option.value = cam.deviceId;
        option.textContent = cam.label || `카메라 ${i + 1}`;
        cameraSelect.appendChild(option);
      });
      cameraSelect.hidden = cameras.length <= 1;
      if (selectedDeviceId && cameras.some((cam) => cam.deviceId === selectedDeviceId)) {
        cameraSelect.value = selectedDeviceId;
      }
    } catch (err) {
      console.warn("카메라 목록을 가져오지 못했습니다:", err.message);
    }
  }

  async function startWebcam() {
    startBtn.disabled = true;
    resultPanel.hidden = true;
    try {
      await ensureModel();
      await ensureBreedModel();

      setStatus("카메라 접근 요청 중...");
      stream = await navigator.mediaDevices.getUserMedia({
        video: getVideoConstraints(),
        audio: false,
      });

      video.srcObject = stream;
      await new Promise((resolve) => {
        video.onloadedmetadata = () => {
          video.play();
          resolve();
        };
      });

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      const trackSettings = stream.getVideoTracks()[0].getSettings();
      mirrored = trackSettings.facingMode !== "environment";
      stage.classList.toggle("mirrored", mirrored);
      selectedDeviceId = trackSettings.deviceId || selectedDeviceId;

      await populateCameraList();

      stage.classList.add("active");
      stopBtn.disabled = false;
      setStatus("인식 중...");
      running = true;
      lastFrameTime = performance.now();
      detectFrame();
    } catch (err) {
      console.error(err);
      startBtn.disabled = false;
      if (err.name === "NotAllowedError") {
        setStatus("카메라 접근이 거부되었습니다. 브라우저 권한 설정을 확인하세요.");
      } else if (err.name === "NotFoundError") {
        setStatus("사용 가능한 카메라를 찾을 수 없습니다.");
      } else {
        setStatus(err.message || "오류가 발생했습니다.");
      }
    }
  }

  function stopScanning() {
    running = false;
    if (rafId) cancelAnimationFrame(rafId);
    rafId = null;

    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      stream = null;
    }
    video.srcObject = null;
    stage.classList.remove("active");
  }

  function stopWebcam() {
    stopScanning();

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    detectionsEl.innerHTML = "";
    fpsEl.textContent = "0";
    lastBreedResults = [];

    startBtn.disabled = false;
    stopBtn.disabled = true;
    setStatus("중지됨");
  }

  async function fetchRepresentativeImage(breedLabelEn) {
    try {
      const res = await fetch(
        `https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(breedLabelEn)}`
      );
      if (!res.ok) return null;
      const data = await res.json();
      return (data.thumbnail && data.thumbnail.source) || (data.originalimage && data.originalimage.source) || null;
    } catch (err) {
      console.warn("대표 이미지를 불러오지 못했습니다:", err.message);
      return null;
    }
  }

  function showBreedResult(result) {
    stopScanning();

    resultImage.removeAttribute("src");
    resultImage.alt = breedLabelKo(result.label);
    resultTitle.textContent = `${classLabelKo(result.class)}: ${breedLabelKo(result.label)}`;
    resultCaption.textContent = `인식 신뢰도 ${Math.round(result.probability * 100)}%`;
    resultPanel.hidden = false;

    startBtn.disabled = false;
    stopBtn.disabled = true;
    setStatus("품종 인식 완료");

    fetchRepresentativeImage(result.label).then((url) => {
      if (url) resultImage.src = url;
    });
  }

  function bboxCenter([x, y, width, height]) {
    return [x + width / 2, y + height / 2];
  }

  function bboxCenterDistance(a, b) {
    const [ax, ay] = bboxCenter(a);
    const [bx, by] = bboxCenter(b);
    return Math.hypot(ax - bx, ay - by);
  }

  function cropRegion(source, [x, y, width, height]) {
    const sx = Math.max(0, Math.round(x));
    const sy = Math.max(0, Math.round(y));
    const sw = Math.max(1, Math.round(Math.min(width, video.videoWidth - sx)));
    const sh = Math.max(1, Math.round(Math.min(height, video.videoHeight - sy)));

    cropCanvas.width = sw;
    cropCanvas.height = sh;
    cropCtx.drawImage(source, sx, sy, sw, sh, 0, 0, sw, sh);
    return cropCanvas;
  }

  async function classifyBreeds(predictions) {
    const results = [];
    let confirmed = null;
    for (const pred of predictions) {
      if (!BreedClassifier.isBreedTarget(pred.class)) continue;
      try {
        const crop = cropRegion(video, pred.bbox);
        const [top] = await BreedClassifier.classify(crop, 1);
        if (top) {
          const result = { class: pred.class, bbox: pred.bbox, label: top.label, probability: top.probability };
          results.push(result);
          if (!confirmed && top.probability >= BREED_CONFIRM_THRESHOLD) {
            confirmed = result;
          }
        }
      } catch (err) {
        console.error("Breed classification error:", err);
      }
    }
    lastBreedResults = results;
    return confirmed;
  }

  function applyCachedBreeds(predictions) {
    predictions.forEach((pred) => {
      if (!BreedClassifier.isBreedTarget(pred.class)) return;
      const [width, height] = [pred.bbox[2], pred.bbox[3]];
      const maxDistance = Math.max(width, height);
      let closest = null;
      let closestDistance = Infinity;
      lastBreedResults.forEach((cached) => {
        if (cached.class !== pred.class) return;
        const distance = bboxCenterDistance(cached.bbox, pred.bbox);
        if (distance < closestDistance) {
          closestDistance = distance;
          closest = cached;
        }
      });
      if (closest && closestDistance <= maxDistance) {
        pred.breedLabel = closest.label;
        pred.breedProbability = closest.probability;
      }
    });
  }

  function drawDetections(predictions) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    predictions.forEach((pred) => {
      const [x, y, width, height] = pred.bbox;
      const label = pred.breedLabel
        ? `${classLabelKo(pred.class)}: ${breedLabelKo(pred.breedLabel)} ${Math.round(pred.breedProbability * 100)}%`
        : `${classLabelKo(pred.class)} ${Math.round(pred.score * 100)}%`;

      ctx.strokeStyle = BOX_COLOR;
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, width, height);

      ctx.font = "16px -apple-system, sans-serif";
      const textWidth = ctx.measureText(label).width;
      ctx.fillStyle = BOX_COLOR;
      ctx.fillRect(x - 1, y - 22, textWidth + 10, 22);

      ctx.fillStyle = "#0f1115";
      if (mirrored) {
        // The canvas is mirrored via CSS (scaleX(-1)) to match the video
        // preview, so text must be flipped back locally or it renders backwards.
        ctx.save();
        ctx.scale(-1, 1);
        ctx.fillText(label, -(x + 5 + textWidth), y - 6);
        ctx.restore();
      } else {
        ctx.fillText(label, x + 5, y - 6);
      }
    });
  }

  function updateDetectionList(predictions) {
    detectionsEl.innerHTML = "";
    const seen = new Map();
    predictions.forEach((pred) => {
      const label = pred.breedLabel
        ? `${classLabelKo(pred.class)}: ${breedLabelKo(pred.breedLabel)}`
        : classLabelKo(pred.class);
      const count = seen.get(label) || 0;
      seen.set(label, count + 1);
    });
    seen.forEach((count, label) => {
      const li = document.createElement("li");
      li.textContent = count > 1 ? `${label} x${count}` : label;
      detectionsEl.appendChild(li);
    });
  }

  async function detectFrame(now = performance.now()) {
    if (!running) return;

    const delta = now - lastFrameTime;
    lastFrameTime = now;
    if (delta > 0) {
      fpsEl.textContent = Math.round(1000 / delta);
    }

    if (now - lastDetectTime >= DETECTION_INTERVAL_MS) {
      lastDetectTime = now;
      try {
        const predictions = await model.detect(video);
        const filtered = predictions.filter((p) => p.score >= MIN_SCORE);

        if (breedReady && now - lastBreedTime >= BREED_INTERVAL_MS) {
          lastBreedTime = now;
          const confirmed = await classifyBreeds(filtered);
          if (confirmed) {
            showBreedResult(confirmed);
            return;
          }
        }
        if (breedReady) {
          applyCachedBreeds(filtered);
        }

        drawDetections(filtered);
        updateDetectionList(filtered);
      } catch (err) {
        console.error("Detection error:", err);
      }
    }

    if (!running) return;
    rafId = requestAnimationFrame(detectFrame);
  }

  startBtn.addEventListener("click", startWebcam);
  stopBtn.addEventListener("click", stopWebcam);
  rescanBtn.addEventListener("click", startWebcam);
  cameraSelect.addEventListener("change", () => {
    selectedDeviceId = cameraSelect.value || null;
    if (running) {
      stopScanning();
      startWebcam();
    }
  });

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setStatus("이 브라우저는 웹캠 접근을 지원하지 않습니다.");
    startBtn.disabled = true;
  } else {
    setStatus("웹캠 시작 버튼을 눌러주세요.");
  }
})();
