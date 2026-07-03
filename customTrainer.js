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

  // classifier.predictClass() only returns a vote fraction per label, not
  // which specific stored example was closest - so which training photo
  // "explains" a match can't be shown from that call alone. similarities()
  // (public on the classifier, used internally by predictClass) returns the
  // raw cosine similarity against every stored example in the same
  // label-insertion order the library concatenates them in, so the single
  // best index can be mapped back to (label, index within that label) using
  // the per-label counts from getClassExampleCount() in that same order.
  async function findNearestExample(embedding) {
    const simsTensor = classifier.similarities(embedding);
    if (!simsTensor) return null;
    const sims = await simsTensor.data();
    simsTensor.dispose();

    let bestIndex = 0;
    let bestSimilarity = -Infinity;
    for (let i = 0; i < sims.length; i++) {
      if (sims[i] > bestSimilarity) {
        bestSimilarity = sims[i];
        bestIndex = i;
      }
    }

    let offset = 0;
    const counts = classifier.getClassExampleCount();
    for (const label of Object.keys(counts)) {
      const count = counts[label];
      if (bestIndex < offset + count) {
        return { label, indexWithinLabel: bestIndex - offset, similarity: bestSimilarity };
      }
      offset += count;
    }
    return null;
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

    // A single class always "wins" 100% of a k-NN vote no matter what the
    // input actually looks like - there's nothing else for it to lose
    // against. Recognition needs at least 2 classes to mean anything.
    canClassify() {
      return !!classifier && classifier.getNumClasses() >= 2;
    },

    async classify(imageElement) {
      if (!this.canClassify()) return null;
      const embedding = embed(imageElement);
      const [result, nearest] = await Promise.all([
        classifier.predictClass(embedding),
        findNearestExample(embedding),
      ]);
      embedding.dispose();
      return { label: result.label, confidence: result.confidences[result.label], nearest };
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
