/*
 * Wires the "나만의 사물 학습하기" panel to CustomTrainer. Deliberately
 * self-contained (reads the #webcam video element and its own controls
 * directly) rather than hooking into app.js's detection loop, so the base
 * COCO-SSD/breed pipeline doesn't need to know this feature exists.
 */
(function () {
  const video = document.getElementById("webcam");
  const startBtn = document.getElementById("start-btn");
  const toggleBtn = document.getElementById("trainer-toggle");
  const body = document.getElementById("trainer-body");
  const addForm = document.getElementById("trainer-add-class-form");
  const titleInput = document.getElementById("trainer-class-title");
  const classList = document.getElementById("trainer-class-list");
  const useToggle = document.getElementById("trainer-use-toggle");
  const exportBtn = document.getElementById("trainer-export-btn");
  const importInput = document.getElementById("trainer-import-input");
  const resetBtn = document.getElementById("trainer-reset-btn");
  const statusEl = document.getElementById("trainer-status");
  const resultEl = document.getElementById("trainer-result");
  const stopBtn = document.getElementById("stop-btn");
  const resultPanel = document.getElementById("result-panel");
  const resultTitle = document.getElementById("result-title");
  const resultImage = document.getElementById("result-image");
  const resultCaption = document.getElementById("result-caption");
  const matchPreview = document.getElementById("trainer-match-preview");
  const matchThumb = document.getElementById("trainer-match-thumb");
  const matchCaption = document.getElementById("trainer-match-caption");
  const candidatesPanel = document.getElementById("trainer-candidates-panel");
  const candidatesList = document.getElementById("trainer-candidates-list");
  const candidatesNoneBtn = document.getElementById("trainer-candidates-none-btn");

  const CLASSIFY_INTERVAL_MS = 600;
  // Once the best candidate looks at least this similar to something
  // trained, pause and ask the user which class it actually is rather than
  // silently picking one - the user's own choice is then fed back in as a
  // new training example for that class to improve future accuracy.
  const CANDIDATE_SIMILARITY_THRESHOLD = 0.7;
  const MAX_CANDIDATES = 3;

  let classifyTimer = null;
  // Ordered per-label arrays (append order matches the order examples were
  // added to the classifier), so a nearest-neighbor index from CustomTrainer
  // can be looked up back to the exact photo that produced it.
  const classThumbnails = {};

  function setStatus(text, isError) {
    statusEl.textContent = text;
    statusEl.classList.toggle("trainer-status-error", !!isError);
  }

  function isWebcamActive() {
    return !!video.srcObject && video.readyState >= 2;
  }

  function snapshotDataUrl() {
    const c = document.createElement("canvas");
    c.width = video.videoWidth;
    c.height = video.videoHeight;
    c.getContext("2d").drawImage(video, 0, 0);
    return c.toDataURL("image/jpeg", 0.7);
  }

  function findClassEl(label) {
    return classList.querySelector(`li[data-label="${CSS.escape(label)}"]`);
  }

  function updateCount(label) {
    const counts = CustomTrainer.getCounts();
    const li = findClassEl(label);
    if (li) {
      li.querySelector(".trainer-class-count").textContent = `${counts[label] || 0}장`;
    }
  }

  function addThumb(label, src) {
    if (!classThumbnails[label]) classThumbnails[label] = [];
    classThumbnails[label].push(src);
    const li = findClassEl(label);
    if (!li) return;
    const img = document.createElement("img");
    img.src = src;
    img.className = "trainer-thumb";
    img.alt = label;
    li.querySelector(".trainer-thumbs").appendChild(img);
  }

  function getThumb(label, index) {
    const thumbs = classThumbnails[label];
    return thumbs && thumbs[index];
  }

  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error("이미지를 불러오지 못했습니다."));
      img.src = src;
    });
  }

  function createClassElement(label) {
    const li = document.createElement("li");
    li.className = "trainer-class";
    li.dataset.label = label;
    li.innerHTML = `
      <div class="trainer-class-header">
        <span class="trainer-class-title"></span>
        <span class="trainer-class-count">0장</span>
        <button type="button" class="trainer-class-remove">삭제</button>
      </div>
      <div class="trainer-class-controls">
        <button type="button" class="trainer-capture-btn">웹캠으로 촬영</button>
        <label class="trainer-upload-label">
          이미지 업로드
          <input type="file" class="trainer-upload-input" accept="image/*" multiple hidden />
        </label>
      </div>
      <div class="trainer-thumbs"></div>
    `;
    li.querySelector(".trainer-class-title").textContent = label;
    classList.appendChild(li);

    li.querySelector(".trainer-class-remove").addEventListener("click", () => {
      CustomTrainer.removeClass(label);
      delete classThumbnails[label];
      li.remove();
      setStatus(`"${label}" 클래스를 삭제했습니다.`);
    });

    li.querySelector(".trainer-capture-btn").addEventListener("click", async () => {
      try {
        if (!isWebcamActive()) {
          setStatus("먼저 웹캠을 시작하세요.", true);
          return;
        }
        setStatus("사진을 등록하는 중... (처음 한 번은 모델을 내려받느라 몇 초 걸릴 수 있습니다)");
        const thumb = snapshotDataUrl();
        await CustomTrainer.addExample(video, label);
        addThumb(label, thumb);
        updateCount(label);
        setStatus(`"${label}"에 사진을 추가했습니다.`);
      } catch (err) {
        console.error("웹캠 촬영 등록 오류:", err);
        setStatus((err && err.message) || "촬영에 실패했습니다. 브라우저 콘솔(F12)에서 자세한 오류를 확인해 주세요.", true);
      }
    });

    li.querySelector(".trainer-upload-input").addEventListener("change", async (e) => {
      const files = Array.from(e.target.files || []);
      e.target.value = "";
      let succeeded = 0;
      for (const file of files) {
        try {
          const img = await loadImage(URL.createObjectURL(file));
          await CustomTrainer.addExample(img, label);
          addThumb(label, img.src);
          updateCount(label);
          succeeded += 1;
        } catch (err) {
          console.error("이미지 등록 실패:", err);
        }
      }
      if (succeeded > 0) {
        setStatus(`"${label}"에 이미지 ${succeeded}장을 등록했습니다.`);
      } else if (files.length) {
        setStatus("이미지 등록에 실패했습니다. 브라우저 콘솔(F12)에서 자세한 오류를 확인해 주세요.", true);
      }
    });

    return li;
  }

  toggleBtn.addEventListener("click", () => {
    const expanded = toggleBtn.getAttribute("aria-expanded") === "true";
    toggleBtn.setAttribute("aria-expanded", String(!expanded));
    body.hidden = expanded;
  });

  addForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const label = titleInput.value.trim();
    if (!label) return;
    if (findClassEl(label)) {
      setStatus("이미 있는 이름입니다.");
      return;
    }
    createClassElement(label);
    titleInput.value = "";
    setStatus(`"${label}" 클래스를 추가했습니다. 사진을 몇 장 등록해 주세요.`);
  });

  function updateMatchPreview(result) {
    const thumb = result && getThumb(result.label, result.nearestIndex);
    if (!thumb) {
      matchPreview.hidden = true;
      return;
    }
    matchThumb.src = thumb;
    matchThumb.alt = result.label;
    matchCaption.textContent = `가장 비슷한 학습 사진: "${result.label}" (유사도 ${Math.round(result.similarity * 100)}%)`;
    matchPreview.hidden = false;
  }

  function showConfirmedResult(candidate, photoSrc, addedToTraining) {
    resultEl.hidden = true;
    matchPreview.hidden = true;
    candidatesPanel.hidden = true;

    resultImage.removeAttribute("src");
    if (photoSrc) resultImage.src = photoSrc;
    resultImage.alt = candidate.label;
    resultTitle.textContent = `나의 모델: ${candidate.label}`;
    resultCaption.textContent = addedToTraining
      ? `직접 선택함 · 이 사진이 "${candidate.label}" 학습 데이터에 추가되어 다음부터 더 정확해집니다.`
      : `인식 신뢰도 ${Math.round(candidate.confidence * 100)}% · 유사도 ${Math.round(candidate.similarity * 100)}%`;
    resultPanel.hidden = false;
  }

  async function pickCandidate(candidate, photoSrc) {
    try {
      const img = await loadImage(photoSrc);
      await CustomTrainer.addExample(img, candidate.label);
      addThumb(candidate.label, photoSrc);
      updateCount(candidate.label);
      showConfirmedResult(candidate, photoSrc, true);
    } catch (err) {
      console.error("정확도 향상을 위한 사진 등록 오류:", err);
      showConfirmedResult(candidate, photoSrc, false);
    }
  }

  function renderCandidateCard(candidate, photoSrc) {
    const card = document.createElement("div");
    card.className = "trainer-candidate";
    const thumb = getThumb(candidate.label, candidate.nearestIndex);
    card.innerHTML = `
      <img class="trainer-candidate-thumb" alt="" />
      <div class="trainer-candidate-info">
        <span class="trainer-candidate-label"></span>
        <span class="trainer-candidate-score"></span>
      </div>
      <button type="button" class="trainer-candidate-pick">이거예요</button>
    `;
    const thumbImg = card.querySelector(".trainer-candidate-thumb");
    if (thumb) thumbImg.src = thumb;
    thumbImg.alt = candidate.label;
    card.querySelector(".trainer-candidate-label").textContent = candidate.label;
    card.querySelector(".trainer-candidate-score").textContent =
      `신뢰도 ${Math.round(candidate.confidence * 100)}% · 유사도 ${Math.round(candidate.similarity * 100)}%`;
    card.querySelector(".trainer-candidate-pick").addEventListener("click", () => {
      pickCandidate(candidate, photoSrc);
    });
    return card;
  }

  function showCandidatePicker(candidates, photoSrc) {
    resultEl.hidden = true;
    matchPreview.hidden = true;
    stopBtn.click();

    candidatesList.innerHTML = "";
    candidates.forEach((candidate) => {
      candidatesList.appendChild(renderCandidateCard(candidate, photoSrc));
    });
    candidatesPanel.hidden = false;
  }

  async function runClassification() {
    if (!isWebcamActive()) return;
    try {
      const candidates = await CustomTrainer.classifyCandidates(video, MAX_CANDIDATES);
      if (!candidates.length) return;
      const top = candidates[0];
      if (top.similarity >= CANDIDATE_SIMILARITY_THRESHOLD) {
        showCandidatePicker(candidates, snapshotDataUrl());
      } else {
        resultEl.textContent =
          `내 모델: 확실하지 않음 (${top.label} 쪽에 가장 가까움 · ` +
          `신뢰도 ${Math.round(top.confidence * 100)}% · 유사도 ${Math.round(top.similarity * 100)}%)`;
        updateMatchPreview(top);
      }
    } catch (err) {
      console.error("커스텀 분류 오류:", err);
    }
  }

  candidatesNoneBtn.addEventListener("click", () => {
    candidatesPanel.hidden = true;
    startBtn.click();
  });

  useToggle.addEventListener("change", async () => {
    if (useToggle.checked) {
      if (!CustomTrainer.canClassify()) {
        setStatus(
          "서로 구분하려면 클래스가 최소 2개 필요합니다. 인식하려는 대상 외에 '배경'처럼 아무것도 아닌 클래스도 하나 추가해 보세요.",
          true
        );
        useToggle.checked = false;
        return;
      }
      resultEl.hidden = false;
      classifyTimer = setInterval(runClassification, CLASSIFY_INTERVAL_MS);
    } else {
      if (classifyTimer) clearInterval(classifyTimer);
      classifyTimer = null;
      resultEl.hidden = true;
      matchPreview.hidden = true;
      candidatesPanel.hidden = true;
    }
  });

  exportBtn.addEventListener("click", async () => {
    try {
      const data = await CustomTrainer.exportDataset();
      if (!data || Object.keys(data).length === 0) {
        setStatus("내보낼 학습 데이터가 없습니다.", true);
        return;
      }
      const blob = new Blob([JSON.stringify(data)], { type: "application/json" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "my-model.json";
      a.click();
      URL.revokeObjectURL(a.href);
      setStatus("모델을 내보냈습니다.");
    } catch (err) {
      console.error("모델 내보내기 오류:", err);
      setStatus("모델을 내보내지 못했습니다.", true);
    }
  });

  importInput.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    e.target.value = "";
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      await CustomTrainer.importDataset(parsed);
      classList.innerHTML = "";
      const counts = CustomTrainer.getCounts();
      Object.keys(counts).forEach((label) => {
        createClassElement(label);
        updateCount(label);
      });
      setStatus("모델을 불러왔습니다.");
    } catch (err) {
      console.error("모델 불러오기 오류:", err);
      setStatus("모델 파일을 불러오지 못했습니다: " + err.message, true);
    }
  });

  resetBtn.addEventListener("click", () => {
    CustomTrainer.reset();
    classList.innerHTML = "";
    Object.keys(classThumbnails).forEach((label) => delete classThumbnails[label]);
    useToggle.checked = false;
    if (classifyTimer) clearInterval(classifyTimer);
    classifyTimer = null;
    resultEl.hidden = true;
    matchPreview.hidden = true;
    candidatesPanel.hidden = true;
    setStatus("전체 초기화했습니다.");
  });
})();
