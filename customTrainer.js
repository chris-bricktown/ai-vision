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

  // classifier.predictClass() only returns an overall vote winner, not a
  // per-class score or which specific stored example was closest - so
  // showing several plausible candidates (each with its own training photo)
  // isn't possible from that call alone. similarities() (public on the
  // classifier, used internally by predictClass) returns the raw cosine
  // similarity against every stored example in the same label-insertion
  // order the library concatenates them in, so per label we can find that
  // label's own best-matching example and rank labels by it.
  async function bestPerLabel(embedding) {
    const simsTensor = classifier.similarities(embedding);
    if (!simsTensor) return [];
    const sims = await simsTensor.data();
    simsTensor.dispose();

    const counts = classifier.getClassExampleCount();
    let offset = 0;
    const results = [];
    for (const label of Object.keys(counts)) {
      const count = counts[label];
      let bestIndex = 0;
      let bestSimilarity = -Infinity;
      for (let i = 0; i < count; i++) {
        const similarity = sims[offset + i];
        if (similarity > bestSimilarity) {
          bestSimilarity = similarity;
          bestIndex = i;
        }
      }
      results.push({ label, indexWithinLabel: bestIndex, similarity: bestSimilarity });
      offset += count;
    }
    return results;
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

    // Ranked candidate classes for the current frame, each scored by its own
    // best-matching training example (not just the overall k-NN vote
    // winner), so label/confidence/similarity/nearestIndex for any one
    // candidate are always about that same class - no mismatch between a
    // displayed name and the training photo shown next to it.
    async classifyCandidates(imageElement, topK = 3) {
      if (!this.canClassify()) return [];
      const embedding = embed(imageElement);
      const [voteResult, perLabel] = await Promise.all([
        classifier.predictClass(embedding),
        bestPerLabel(embedding),
      ]);
      embedding.dispose();
      return perLabel
        .map((entry) => ({
          label: entry.label,
          confidence: voteResult.confidences[entry.label] || 0,
          similarity: entry.similarity,
          nearestIndex: entry.indexWithinLabel,
        }))
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, topK);
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
