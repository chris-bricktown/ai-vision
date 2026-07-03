/*
 * Dog/cat breed classifier. Wraps whichever model backend is active behind
 * a fixed interface (load / classify) so the MobileNet backend used today
 * can be swapped for a custom-trained model without touching app.js.
 *
 * `customBackend` below is a working example of the swap: a small model
 * trained on 6 breeds and converted with tensorflowjs_converter (see
 * README's "커스텀 모델 학습" section). It's a proof of concept, not a
 * production replacement — MobileNet covers far more breeds. To try it,
 * change `activeBackend` at the bottom of this file.
 *
 * To add a different custom backend: implement an object with the same
 * shape as `mobilenetBackend`/`customBackend` (load() and
 * classify(imageElement, topK, cocoClass)) and point `activeBackend` at it.
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
    // MobileNet is a general 1000-class ImageNet classifier, not a
    // dog/cat-breed-only model - its raw top predictions can land on a
    // completely unrelated class (e.g. "ice bear") even when COCO-SSD
    // already correctly identified the crop as a cat. Ask for every class's
    // probability and keep only ones that are an actual known breed for the
    // detected category (labels_ko.js's DOG_BREED_LABELS_KO/
    // CAT_BREED_LABELS_KO - the same lists used to translate a breed name
    // are also the authoritative set of what counts as a valid one), so an
    // off-category guess never surfaces as if it were a real breed match.
    async classify(imageElement, topK, cocoClass) {
      const allPredictions = await this.model.classify(imageElement, 1000);
      const validLabels =
        cocoClass === "dog" ? global.DOG_BREED_LABELS_KO : cocoClass === "cat" ? global.CAT_BREED_LABELS_KO : null;
      return allPredictions
        .map((p) => ({ label: p.className.split(",")[0].trim(), probability: p.probability }))
        .filter((p) => !validLabels || Object.prototype.hasOwnProperty.call(validLabels, p.label))
        .slice(0, topK);
    },
  };

  // Small proof-of-concept custom model: MobileNetV2 (frozen, ImageNet
  // weights) fine-tuned on 6 classes (3 cat + 3 dog breeds) from the
  // Oxford-IIIT Pet Dataset. Demonstrates the training -> tensorflowjs_converter
  // -> tf.loadLayersModel pipeline described above; not a production
  // replacement for the much broader MobileNet/ImageNet backend.
  const CUSTOM_MODEL_URL = "models/custom-breeds/model.json";
  const CUSTOM_LABELS = ["Persian", "Siamese", "Bengal", "beagle", "pug", "yorkshire_terrier"];
  const CUSTOM_INPUT_SIZE = 224;

  const customBackend = {
    model: null,
    async load() {
      if (this.model) return this.model;
      if (typeof tf === "undefined") {
        throw new Error("TensorFlow.js를 불러오지 못했습니다.");
      }
      this.model = await tf.loadLayersModel(CUSTOM_MODEL_URL);
      return this.model;
    },
    async classify(imageElement, topK, _cocoClass) {
      const output = tf.tidy(() => {
        // Matches the Python-side preprocessing applied before training
        // (kept out of the model graph itself; see train.py) — rescale
        // [0, 255] pixels to [-1, 1] the way MobileNetV2 expects.
        const input = tf.browser
          .fromPixels(imageElement)
          .resizeBilinear([CUSTOM_INPUT_SIZE, CUSTOM_INPUT_SIZE])
          .toFloat()
          .div(127.5)
          .sub(1)
          .expandDims(0);
        return this.model.predict(input);
      });
      const probabilities = await output.data();
      output.dispose();
      return Array.from(probabilities)
        .map((probability, index) => ({ label: CUSTOM_LABELS[index], probability }))
        .sort((a, b) => b.probability - a.probability)
        .slice(0, topK);
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
    classify(imageElement, topK = 1, cocoClass) {
      return activeBackend.classify(imageElement, topK, cocoClass);
    },
  };
})(window);
