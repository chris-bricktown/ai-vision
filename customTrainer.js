/*
 * In-browser "teach it yourself" classifier: the user adds a class by
 * title and feeds it a few images (webcam capture or file upload). There's
 * no separate training step with epochs - a k-NN classifier on top of
 * MobileNet's penultimate-layer embeddings (the same transfer-learning
 * trick Teachable Machine uses) treats each added image as an example
 * immediately, so results are usable after just a handful of photos.
 */
(function (global) {
  let classifier = null;
  let embeddingModel = null;
  let ready = false;

  async function ensureReady() {
    if (ready) return;
    if (typeof knnClassifier === "undefined" || typeof mobilenet === "undefined") {
      throw new Error("학습에 필요한 라이브러리를 불러오지 못했습니다.");
    }
    classifier = knnClassifier.create();
    embeddingModel = await mobilenet.load({ version: 2, alpha: 1.0 });
    ready = true;
  }

  function embed(imageElement) {
    return tf.tidy(() => embeddingModel.infer(imageElement, true));
  }

  global.CustomTrainer = {
    async addExample(imageElement, label) {
      await ensureReady();
      const embedding = embed(imageElement);
      classifier.addExample(embedding, label);
      embedding.dispose();
    },

    removeClass(label) {
      if (classifier) classifier.clearClass(label);
    },

    getCounts() {
      return classifier ? classifier.getClassExampleCount() : {};
    },

    classCount() {
      return classifier ? classifier.getNumClasses() : 0;
    },

    async classify(imageElement) {
      if (!classifier || classifier.getNumClasses() === 0) return null;
      const embedding = embed(imageElement);
      const result = await classifier.predictClass(embedding);
      embedding.dispose();
      return { label: result.label, confidence: result.confidences[result.label] };
    },

    async exportDataset() {
      if (!classifier) return {};
      const dataset = classifier.getClassifierDataset();
      const serialized = {};
      for (const label of Object.keys(dataset)) {
        const tensor = dataset[label];
        serialized[label] = { data: Array.from(await tensor.data()), shape: tensor.shape };
      }
      return serialized;
    },

    async importDataset(serialized) {
      await ensureReady();
      const dataset = {};
      Object.keys(serialized).forEach((label) => {
        dataset[label] = tf.tensor(serialized[label].data, serialized[label].shape);
      });
      classifier.setClassifierDataset(dataset);
    },

    reset() {
      if (classifier) classifier.clearAllClasses();
    },
  };
})(window);
