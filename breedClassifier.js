/*
 * Dog/cat breed classifier. Wraps whichever model backend is active behind
 * a fixed interface (load / classify) so the backend(s) used today can be
 * changed without touching app.js.
 *
 * The default `activeBackend` (combinedBackend, below) runs both
 * `mobilenetBackend` (ImageNet, ~120 dog breeds but only 5 cat breeds) and
 * `customBackend` (a model trained on all 37 Oxford-IIIT Pet Dataset
 * breeds - see README's "커스텀 모델 학습" section - 12 cat breeds, 25 dog
 * breeds) on every crop and keeps whichever is more confident, so neither
 * backend's weak spot (MobileNet on cats, the custom model on dog breeds
 * outside its 25) dominates the result.
 *
 * To add a different backend: implement an object with the same shape
 * (load() and classify(imageElement, topK, cocoClass)) and point
 * `activeBackend` at it, or fold it into combinedBackend the same way.
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

  // Custom model: MobileNetV2 (frozen, ImageNet weights) fine-tuned on all
  // 37 breeds (12 cat + 25 dog) in the Oxford-IIIT Pet Dataset, converted
  // with tensorflowjs_converter. Demonstrates the training ->
  // tensorflowjs_converter -> tf.loadLayersModel pipeline (see
  // training/README.md); covers every cat breed the dataset has (more than
  // MobileNet/ImageNet's 5), though fewer dog breeds than ImageNet's ~120.
  // CUSTOM_LABELS order must exactly match train.py's CLASSES (the model's
  // output layer is indexed by that order).
  const CUSTOM_MODEL_URL = "models/custom-breeds/model.json";
  // CUSTOM_LABELS order must exactly match train.py's CLASSES (the model's
  // output layer is indexed by that order): 12 cats then 25 dogs.
  const CUSTOM_CAT_LABELS = [
    "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
    "Egyptian_Mau", "Maine_Coon", "Persian", "Ragdoll", "Russian_Blue",
    "Siamese", "Sphynx",
  ];
  const CUSTOM_DOG_LABELS = [
    "american_bulldog", "american_pit_bull_terrier", "basset_hound",
    "beagle", "boxer", "chihuahua", "english_cocker_spaniel",
    "english_setter", "german_shorthaired", "great_pyrenees", "havanese",
    "japanese_chin", "keeshond", "leonberger", "miniature_pinscher",
    "newfoundland", "pomeranian", "pug", "saint_bernard", "samoyed",
    "scottish_terrier", "shiba_inu", "staffordshire_bull_terrier",
    "wheaten_terrier", "yorkshire_terrier",
  ];
  const CUSTOM_LABELS = [...CUSTOM_CAT_LABELS, ...CUSTOM_DOG_LABELS];
  const CUSTOM_CAT_SET = new Set(CUSTOM_CAT_LABELS);
  const CUSTOM_DOG_SET = new Set(CUSTOM_DOG_LABELS);
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
    async classify(imageElement, topK, cocoClass) {
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
      // Keep only breeds matching the category COCO-SSD already detected -
      // otherwise a dog crop can surface a high-confidence cat breed (e.g.
      // "Bengal") since this 37-class model has no notion of dog-vs-cat on
      // its own, producing a "개: 벵골 고양이" style mismatch.
      const validSet = cocoClass === "dog" ? CUSTOM_DOG_SET : cocoClass === "cat" ? CUSTOM_CAT_SET : null;
      return Array.from(probabilities)
        .map((probability, index) => ({ label: CUSTOM_LABELS[index], probability }))
        .filter((p) => !validSet || validSet.has(p.label))
        .sort((a, b) => b.probability - a.probability)
        .slice(0, topK);
    },
  };

  // Runs both backends on every crop and keeps whichever prediction is more
  // confident: MobileNet covers far more dog breeds (~120 vs 25), the
  // custom model covers more cat breeds (12 vs 5) and was trained
  // specifically to avoid off-category guesses, so neither backend alone is
  // strictly better. Note the two probabilities aren't perfectly
  // comparable (a 37-class softmax tends to peak higher than a 1000-class
  // one for the same underlying confidence), but comparing them directly is
  // still a reasonable, simple way to combine two otherwise-independent
  // opinions on the same crop.
  const combinedBackend = {
    async load() {
      await Promise.all([mobilenetBackend.load(), customBackend.load()]);
    },
    async classify(imageElement, topK, cocoClass) {
      const [mobilenetResults, customResults] = await Promise.all([
        mobilenetBackend.classify(imageElement, topK, cocoClass),
        customBackend.classify(imageElement, topK, cocoClass),
      ]);
      return [...mobilenetResults, ...customResults]
        .sort((a, b) => b.probability - a.probability)
        .slice(0, topK);
    },
  };

  const activeBackend = combinedBackend;

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
