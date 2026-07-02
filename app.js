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

  const DETECTION_INTERVAL_MS = 100;
  const MIN_SCORE = 0.5;
  const BOX_COLOR = "#37e07a";

  let model = null;
  let stream = null;
  let running = false;
  let rafId = null;
  let lastDetectTime = 0;
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

  async function startWebcam() {
    startBtn.disabled = true;
    try {
      await ensureModel();

      setStatus("카메라 접근 요청 중...");
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
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

  function stopWebcam() {
    running = false;
    if (rafId) cancelAnimationFrame(rafId);
    rafId = null;

    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      stream = null;
    }
    video.srcObject = null;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    detectionsEl.innerHTML = "";
    fpsEl.textContent = "0";

    stage.classList.remove("active");
    startBtn.disabled = false;
    stopBtn.disabled = true;
    setStatus("중지됨");
  }

  function drawDetections(predictions) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    predictions.forEach((pred) => {
      const [x, y, width, height] = pred.bbox;
      const label = `${pred.class} ${Math.round(pred.score * 100)}%`;

      ctx.strokeStyle = BOX_COLOR;
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, width, height);

      ctx.font = "16px -apple-system, sans-serif";
      const textWidth = ctx.measureText(label).width;
      ctx.fillStyle = BOX_COLOR;
      ctx.fillRect(x - 1, y - 22, textWidth + 10, 22);

      // The canvas is mirrored via CSS (scaleX(-1)) to match the video preview,
      // so text must be flipped back locally or it renders backwards.
      ctx.save();
      ctx.scale(-1, 1);
      ctx.fillStyle = "#0f1115";
      ctx.fillText(label, -(x + 5 + textWidth), y - 6);
      ctx.restore();
    });
  }

  function updateDetectionList(predictions) {
    detectionsEl.innerHTML = "";
    const seen = new Map();
    predictions.forEach((pred) => {
      const count = seen.get(pred.class) || 0;
      seen.set(pred.class, count + 1);
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
        drawDetections(filtered);
        updateDetectionList(filtered);
      } catch (err) {
        console.error("Detection error:", err);
      }
    }

    rafId = requestAnimationFrame(detectFrame);
  }

  startBtn.addEventListener("click", startWebcam);
  stopBtn.addEventListener("click", stopWebcam);

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setStatus("이 브라우저는 웹캠 접근을 지원하지 않습니다.");
    startBtn.disabled = true;
  } else {
    setStatus("웹캠 시작 버튼을 눌러주세요.");
  }
})();
