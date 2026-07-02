/*
 * Dog/cat breed classifier. Wraps whichever model backend is active behind
 * a fixed interface (load / classify) so the MobileNet backend used today
 * can be swapped for a custom-trained model later without touching app.js.
 *
 * To add a custom backend once a trained model exists:
 *   1. Convert it to TF.js format (tensorflowjs_converter) and host the
 *      model.json + weight shards as static files (e.g. under /models/breeds/).
 *   2. Implement an object with the same shape as `mobilenetBackend` below
 *      (load() and classify(imageElement, topK)) that loads the model with
 *      tf.loadGraphModel/tf.loadLayersModel and maps output logits to your
 *      label list.
 *   3. Point `activeBackend` at it.
 */
(function (global) {
  const BREED_TARGET_CLASSES = new Set(["dog", "cat"]);

  const mobilenetBackend = {
    model: null,
    async load() {
      if (this.model) return this.model;
      if (typeof mobilenet === "undefined") {
        throw new Error("MobileNet 라이브러리를 불러오지 못했습니다.");
      }
      this.model = await mobilenet.load({ version: 2, alpha: 1.0 });
      return this.model;
    },
    async classify(imageElement, topK) {
      const predictions = await this.model.classify(imageElement, topK);
      return predictions.map((p) => ({
        label: p.className.split(",")[0].trim(),
        probability: p.probability,
      }));
    },
  };

  const activeBackend = mobilenetBackend;

  global.BreedClassifier = {
    isBreedTarget(cocoClassName) {
      return BREED_TARGET_CLASSES.has(cocoClassName);
    },
    load() {
      return activeBackend.load();
    },
    classify(imageElement, topK = 1) {
      return activeBackend.classify(imageElement, topK);
    },
  };
})(window);
