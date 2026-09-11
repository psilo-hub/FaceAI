# FaceAI — Java Face Detection & Recognition Library

## 0. How to Use This Document

This file is the source of truth for OpenCode. **Status: Phases 0–8 are complete and released as `0.1.0`.** The remaining work is optional Phase 9. When changing behavior or adding public API, update this document in the same change. Use the phase checklists below to track what is already done.

---

## 1. Summary

Build a small Java library for face detection and face recognition using DJL.

The library operates on `java.awt.image.BufferedImage` and exposes four main operations:

1. Detect faces in an image.
2. Compute a face embedding from a cropped face image.
3. Compute similarity between two embeddings.
4. Compute an average embedding from a list of embeddings.

Model choices are fixed:

- Face detection: **RetinaFace**
- Face recognition: **FaceNet** exported from **facenet-pytorch**

Models must be downloaded automatically on first use and cached locally. The library must not require Python at runtime.

---

## 2. Goals and Non-Goals

### Goals

- Simple public API.
- Works with `BufferedImage`.
- Uses DJL and the PyTorch engine.
- Auto-downloads RetinaFace and FaceNet models on first use.
- Returns bounding boxes, confidence scores, and embeddings.
- Provides cosine-based similarity mapped to `[0.0, 1.0]`.
- Provides arithmetic average of embeddings.
- Unit-testable utility methods.
- Integration-testable with real model files.

### Non-Goals

- Training or fine-tuning models.
- Face alignment.
- Anti-spoofing / liveness detection.
- Video processing.
- GUI or web service.
- Android support.
- Multiple selectable detection/recognition models.
- Runtime Python dependency.

---

## 3. Public API

Package root: `free.svoss.tools.faceai`

Internal implementation package: `free.svoss.tools.faceai.internal`

### 3.1 `FaceAI`

Main facade. Holds loaded models. Not thread-safe unless documented otherwise.

```java
public final class FaceAI implements AutoCloseable {
    public static FaceAI create();
    public static FaceAI create(FaceAIConfig config);

    public DetectedFace[] detectFaces(BufferedImage image);
    public float[] getEmbedding(BufferedImage faceImage);
    public double calcSimilarity(float[] embeddingA, float[] embeddingB);
    public float[] calcAverage(List<float[]> embeddings);

    public void close();
}
```

### 3.2 `DetectedFace`

Value object for one detected face.

```java
public final class DetectedFace {
    public int x();
    public int y();
    public int width();
    public int height();
    public float confidence();

    public Rectangle bounds();
    public BufferedImage crop(BufferedImage source);
}
```

Requirements:

- `x`, `y` are top-left coordinates in pixels.
- `width`, `height` are in pixels.
- Coordinates are clamped to the source image bounds.
- `confidence` is between `0.0` and `1.0`.
- `crop(source)` returns a sub-image for the bounding box.
- `equals`, `hashCode`, and `toString` should be implemented.
- Optional future field: 5-point landmarks. Not required for v1.

### 3.3 `FaceAIConfig`

Configuration object with defaults.

Fields:

- `cacheDir`: default `~/.djl.ai/cache` or respect `DJL_CACHE_DIR`.
- `detectionThreshold`: default `0.8`.
- `nmsThreshold`: default `0.4`.
- `embeddingDimension`: default `512`.
- `device`: default CPU; optional GPU if available.
- `l2NormalizeEmbeddings`: default `true`.
- `detectionModelUrl`: optional override.
- `recognitionModelUrl`: optional override.

Provide a builder.

### 3.4 `FaceAIException`

Runtime exception for all library-specific failures.

- Must wrap DJL exceptions.
- Must not leak DJL types into public API signatures.
- Should include clear messages for model download failures, missing cache, invalid input, etc.

### 3.5 Method Contracts

#### `detectFaces(BufferedImage image)`

- Input must be non-null and have positive dimensions.
- Converts input to RGB internally.
- Applies RetinaFace preprocessing.
- Applies confidence threshold.
- Applies non-maximum suppression.
- Returns `DetectedFace[]`, sorted by descending confidence.
- Returns an empty array if no faces are found. Never returns `null`.
- Does not modify the input image.

#### `getEmbedding(BufferedImage faceImage)`

- Input must be non-null and have positive dimensions.
- Assumes the image contains exactly one face crop. It does not run detection.
- Converts to RGB.
- Resizes to `160x160`.
- Applies FaceNet preprocessing matching `facenet-pytorch`.
- Runs the model.
- Returns a `float[]` of length `512`.
- If `l2NormalizeEmbeddings` is true, returns an L2-normalized vector.
- Throws `FaceAIException` if preprocessing or inference fails.

#### `calcSimilarity(float[] embeddingA, float[] embeddingB)`

- Both inputs must be non-null.
- Both must have the same length.
- Both must have non-zero norm.
- Normalizes inputs internally if needed.
- Computes cosine similarity.
- Maps cosine from `[-1, 1]` to `[0, 1]` using `(cos + 1.0) / 2.0`.
- Returns `1.0` for identical direction.
- Returns `0.5` for orthogonal vectors.
- Returns `0.0` for opposite direction.
- Clamps to `[0.0, 1.0]` to avoid floating-point drift.
- Throws `IllegalArgumentException` for invalid inputs.

#### `calcAverage(List<float[]> embeddings)`

- List must be non-null and non-empty.
- No element may be null.
- All embeddings must have the same length.
- Returns element-wise arithmetic mean.
- Result length equals input embedding length.
- Does not L2-normalize the result. `calcSimilarity` normalizes internally.
- Optional future method: `calcMedian`. Not required for v1.

---

## 4. Models

### 4.1 Detection: RetinaFace

- Use RetinaFace via DJL PyTorch engine.
- Implemented as a TorchScript export loaded from the DJL test-models repo:
  `https://resources.djl.ai/test-models/pytorch/retinaface.zip` (model file `retinaface.pt`). URL overridable via `FaceAIConfig.detectionModelUrl`.
- Output format: three tensors — `loc` `[N,4]` anchor offsets, `conf` `[N,2]` `[background, face]`, `landms` `[N,10]` landmarks. Boxes are decoded from anchor priors (RetinaFace/ResNet-50 config: min sizes `[[16,32],[64,128],[256,512]]`, strides `[8,16,32]`, variance `[0.1,0.2]`).
- Required post-processing (implemented in `RetinaFaceTranslator`):
  - Decode anchor offsets to boxes (normalized) and map to pixel coordinates.
  - Confidence threshold (default `0.8`), keep top-K (`5000`).
  - Non-maximum suppression with IoU threshold `0.4`.
  - Clamp boxes to image bounds.
  - Sort by confidence descending.
- Input: runs at native resolution — BGR channels, values `[0,255]`, minus BGR mean `[104,117,123]`, CHW `3×H×W` (DJL adds the leading batch dim). No letterbox/resize is applied; priors are generated for the input size.

### 4.2 Recognition: FaceNet via facenet-pytorch

- Use FaceNet exported from `facenet-pytorch` to TorchScript. The Java library must not run Python.
- Implemented model: `InceptionResnetV1(pretrained='vggface2').eval()` export from the DJL test-models repo:
  `https://resources.djl.ai/test-models/pytorch/face_feature.zip` (model file `face_feature.pt`). URL overridable via `FaceAIConfig.recognitionModelUrl`.
- Input: `1x3x160x160` RGB tensor.
- Output: `1x512` embedding.
- Preprocessing matches `facenet-pytorch` exactly: `(pixel/255 − 0.5) / 0.5` → `[-1, 1]`. Documented in `FaceNetTranslator` and `ModelConstants`.
- L2-normalize the embedding before returning when `l2NormalizeEmbeddings=true` (default).

### 4.3 Auto-Download and Cache

- Use DJL `Criteria` with `optModelUrls` or `optModelName`.
- DJL should download models on first use.
- Cache location:
  - Respect `FaceAIConfig.cacheDir` if set.
  - Otherwise respect `DJL_CACHE_DIR`.
  - Otherwise use `~/.djl.ai/cache`.
- First call may be slow. Subsequent calls must use the cache.
- If models are missing and download fails, throw `FaceAIException` with the URL and cache path.
- Log model download and cache usage with SLF4J.

---

## 5. Architecture

Internal classes (all in `free.svoss.tools.faceai.internal`):

- `RetinaFaceDetector`
  - Owns the detection model + `Predictor<BufferedImage, DetectedFace[]>`.
  - Builds a DJL `Criteria` with the RetinaFace URL, loads via `ModelLoader`.
  - Accepts an injected predictor for testing.
- `RetinaFaceTranslator`
  - Implements `Translator<BufferedImage, DetectedFace[]>` directly.
  - Preprocessing: BGR channels at native resolution, values `[0,255]`, minus BGR mean `[104,117,123]`, CHW `3×H×W` (batch dim added by DJL). Mirrors `Pytorch_Retinaface`.
  - Post-processing: decodes `loc`/`conf` tensors via generated anchor priors (`generatePriors`, package-private static), applies the confidence threshold, keeps top-K, clamps to input bounds, applies NMS (`applyNms`/`computeIoU`, package-private static for testing), sorts by confidence descending.
- `FaceNetRecognizer`
  - Owns the recognition model + `Predictor<BufferedImage, float[]>`.
  - Builds a DJL `Criteria` with the FaceNet URL, loads via `ModelLoader`.
  - Verifies the returned embedding length matches `config.embeddingDimension()`.
- `FaceNetTranslator`
  - Implements `Translator<BufferedImage, float[]>` directly.
  - Preprocessing: RGB, resize to `160×160`, `(pixel/255 − 0.5) / 0.5` (documented in Javadoc), NCHW `1×3×160×160`.
  - L2-normalizes the output embedding when `config.l2NormalizeEmbeddings()` is set.
- `FaceDetectionTranslator`
  - Legacy/unused class (anchor-decoding RetinaFace `Translator<Image, ...>` from an earlier iteration). Not referenced by the current pipeline; kept in the repo. **Remove in a future cleanup.**
- `ImageUtils`
  - RGB conversion (alpha compositing over white), resizing (bilinear), letterbox (gray `128,128,128` padding), `rgbPixels`, `toNDArray` (`[0,1]`), `toNDArrayNormalized` (FaceNet `[-1,1]`). Pure Java, no DJL types in signatures.
- `PostProcessing`
  - Pure-Java utilities: `process` (threshold + NMS + clamp + sort), `nms`, `computeIoU`, `decodeBoxes`. Unit tested independently; the active RetinaFace path uses `RetinaFaceTranslator`'s own NMS helpers instead.
- `EmbeddingUtils`
  - Cosine similarity mapped to `[0,1]`, element-wise arithmetic mean. Pure Java.
- `ModelLoader`
  - Builds DJL `Criteria`, handles cache dir (sets `DJL_CACHE_DIR` system property when `config.cacheDir()` is set), logs download URL and cache path, wraps DJL exceptions in `FaceAIException`.
- `ModelConstants`
  - Default model URLs (DJL test-models repo), model file names, engine name, FaceNet preprocessing constants, RetinaFace input size.

Public API must not expose DJL types.

`FaceAI` lazy-loads models on first use (double-checked locking on volatile fields) and closes predictors/models in `close()`.

Thread safety:

- `FaceAI` is not thread-safe by default, but lazy loading is guarded so concurrent first-use won't double-load.
- Document that each thread should use its own instance or external synchronization.
- Optional Phase 9: thread-local predictors or synchronized wrappers.

---

## 6. Implementation Phases

### Phase 0 — Project Skeleton ✅

- [x] Create Maven project with Java 11.
- [x] Add dependencies:
  - [x] `ai.djl:api`
  - [x] `ai.djl.pytorch:pytorch-engine`
  - [x] `ai.djl.pytorch:pytorch-native-cpu` (runtime scope) — **note:** `pytorch-native-auto` does not exist for DJL 0.31.x; the cpu auto-selector jar plus `pytorch-jni` is used instead
  - [x] `org.slf4j:slf4j-api`
  - [x] `org.slf4j:slf4j-simple` for tests
  - [x] JUnit 5
- [x] Create packages:
  - [x] `free.svoss.tools.faceai`
  - [x] `free.svoss.tools.faceai.internal`
- [x] Add `.gitignore`.
- [x] Add `README.md`.
- [x] Add placeholder tests.

Deliverable: `mvn test` passes with no model loading.

Acceptance: project compiles, tests run, dependency tree is valid.

---

### Phase 1 — Public API and Data Types ✅

- [x] Implement `DetectedFace`.
- [x] Implement `FaceAIConfig` with builder and defaults.
- [x] Implement `FaceAIException`.
- [x] Create `FaceAI` skeleton with method signatures and Javadoc.
- [x] Add unit tests for `DetectedFace`:
  - [x] `bounds()`
  - [x] `crop()`
  - [x] `equals` / `hashCode`
- [x] Add unit tests for `FaceAIConfig` defaults.

Deliverable: public API compiles.

Acceptance: no DJL types appear in public signatures. Unit tests pass.

---

### Phase 2 — Model Loading and Auto-Download ✅

- [x] Implement `ModelLoader`.
- [x] Define model constants in `ModelConstants` for RetinaFace and FaceNet.
- [x] Use DJL `Criteria`.
- [x] Support `FaceAIConfig.cacheDir`.
- [x] Respect `DJL_CACHE_DIR`.
- [x] Log download URL and cache path.
- [x] Add an integration test tagged `integration` that loads both models (via `FaceAIIntegrationTest`).
- [x] Ensure second run works offline from cache.

Deliverable: models load without manual download.

Acceptance: first run downloads; second run uses cache. Failures throw `FaceAIException` with clear cause.

---

### Phase 3 — Face Detection with RetinaFace ✅

- [x] Implement `RetinaFaceDetector`.
- [x] Implement `RetinaFaceTranslator`.
- [x] Implement RGB conversion.
- [x] Run inference (native resolution, BGR mean subtraction — resize/letterbox not required for this export).
- [x] Run inference.
- [x] Apply confidence threshold.
- [x] Decode boxes if required by model output (the current 2-D output format needs no anchor decoding; `PostProcessing.decodeBoxes` exists for anchor-based outputs).
- [x] Apply NMS with IoU threshold `0.4`.
- [x] Clamp boxes to image bounds.
- [x] Sort detections by confidence descending.
- [x] Add integration test with a sample face image.
- [x] Assert at least one face is detected.
- [x] Assert confidence is above threshold.
- [x] Assert bounds are inside the image.

Deliverable: `detectFaces` works on real images.

Acceptance: integration test passes. Empty image returns empty array, not `null`.

---

### Phase 4 — Face Recognition Embedding with FaceNet ✅

- [x] Implement `FaceNetRecognizer`.
- [x] Implement `FaceNetTranslator`.
- [x] Implement preprocessing:
  - [x] RGB conversion.
  - [x] Resize to `160x160`.
  - [x] Match `facenet-pytorch` normalization exactly (`(pixel/255 − 0.5)/0.5`).
- [x] Run inference.
- [x] Extract `float[512]`.
- [x] L2-normalize if configured.
- [x] Add integration test:
  - [x] Detect a face.
  - [x] Crop it.
  - [x] Get embedding.
  - [x] Assert length is `512`.
  - [x] Assert norm is approximately `1.0` if normalization enabled.
  - [x] Assert same identity similarity is higher than different identity similarity.

Deliverable: `getEmbedding` works on face crops.

Acceptance: integration test passes. Preprocessing is documented in Javadoc.

---

### Phase 5 — Similarity and Average ✅

- [x] Implement `EmbeddingUtils`.
- [x] Implement `calcSimilarity`:
  - [x] Validate inputs.
  - [x] Normalize internally.
  - [x] Compute cosine.
  - [x] Map to `[0,1]`.
  - [x] Clamp.
- [x] Implement `calcAverage`:
  - [x] Validate list.
  - [x] Validate equal dimensions.
  - [x] Compute element-wise mean.
- [x] Unit tests:
  - [x] Identical vectors return `1.0`.
  - [x] Orthogonal vectors return `0.5`.
  - [x] Opposite vectors return `0.0`.
  - [x] Zero vector throws.
  - [x] Dimension mismatch throws.
  - [x] Average of `[1,0]` and `[0,1]` returns `[0.5,0.5]`.
  - [x] Empty list throws.

Deliverable: utility methods fully tested.

Acceptance: all unit tests pass.

---

### Phase 6 — Facade Integration ✅

- [x] Implement `FaceAI.create()` and `FaceAI.create(FaceAIConfig)`.
- [x] Lazy-load models on first use.
- [x] Wire `detectFaces` to `RetinaFaceDetector`.
- [x] Wire `getEmbedding` to `FaceNetRecognizer`.
- [x] Wire `calcSimilarity` and `calcAverage` to `EmbeddingUtils`.
- [x] Implement `close()`.
- [x] Add input validation with clear exceptions.
- [x] Add end-to-end integration test:
  - [x] Load image.
  - [x] Detect faces.
  - [x] Crop first face.
  - [x] Get embedding.
  - [x] Compare with another embedding.
  - [x] Compute average of two embeddings.

Deliverable: all four public methods work end-to-end.

Acceptance: end-to-end test passes. No DJL types leak to public API.

---

### Phase 7 — Testing and Validation ✅

- [x] Add sample images under `src/test/resources/faces/`:
  - [x] `person1_a.jpg`
  - [x] `person1_b.jpg`
  - [x] `person2.jpg`
- [x] Add integration test profile in Maven.
- [x] Tag model-dependent tests with `@Tag("integration")`.
- [x] Add edge-case tests:
  - [x] Null image.
  - [x] Empty image.
  - [x] Null embeddings.
  - [x] Wrong embedding dimension.
  - [x] Zero vector.
  - [x] Empty list for average.
- [x] Add model download/cache test.
- [x] Ensure unit tests run without network.

Deliverable: robust test suite (82 unit tests + integration suite).

Acceptance: `mvn test` runs unit tests. `mvn verify -Pintegration` runs integration tests.

---

### Phase 8 — Documentation and Release ✅

- [x] Write `README.md`:
  - [x] Installation.
  - [x] Quickstart.
  - [x] Model download and cache behavior.
  - [x] API examples.
  - [x] Limitations.
  - [x] Thread-safety note.
- [x] Add Javadoc for all public types and methods.
- [x] Add `CHANGELOG.md`.
- [x] Set version to `0.1.0`.
- [x] Add license file (`LICENSE`, MIT).

Deliverable: library is usable by a new developer.

Acceptance: README contains a copy-paste example that compiles. Javadoc builds without errors.

---

### Phase 9 — Optional Enhancements

Do not start until Phases 0–8 are complete.

- [ ] GPU support via DJL device selection.
- [ ] Batch detection and batch embedding.
- [ ] `calcMedian(List<float[]>)`.
- [ ] 5-point landmarks in `DetectedFace`.
- [ ] Face alignment using landmarks.
- [ ] Configurable similarity threshold helper.
- [ ] Thread-safe wrapper.
- [ ] Additional model backends.
- [ ] Anti-spoofing hooks.

---

## 7. Testing Strategy

- Unit tests (run by `mvn test`, no network/model needed):
  - `DetectedFaceTest`
  - `FaceAIConfigTest`
  - `FaceAITest` (facade behavior, validation, closed-instance)
  - `EmbeddingUtilsTest`
  - `ImageUtilsTest`
  - `PostProcessingTest`
  - `RetinaFaceTranslatorTest` (NMS + IoU helpers)
  - `FaceNetTranslatorTest`
  - `ModelLoaderTest` (constants + cache-dir resolution)
- Integration tests (tagged `@Tag("integration")`, run by `mvn verify -Pintegration`):
  - `FaceAIIntegrationTest` — end-to-end detect → crop → embed → compare using the bundled sample faces.
- Test images:
  - `src/test/resources/faces/person1_a.jpg`
  - `src/test/resources/faces/person1_b.jpg`
  - `src/test/resources/faces/person2.jpg`
- Assertions:
  - Same-person similarity > different-person similarity.
  - Detection confidence > threshold.
  - Bounding boxes within image bounds.
  - Embedding length is `512`.
  - Embedding norm is approximately `1.0` when normalization is enabled.
- Integration tests may be slow on first run due to model download. Tag them and document.

---

## 8. Whole-Project Acceptance Criteria

- [x] All four public methods implemented.
- [x] Models auto-download on first use.
- [x] Subsequent runs work from cache.
- [x] Public API has no DJL types.
- [x] `detectFaces` returns empty array when no face is found.
- [x] `getEmbedding` returns `float[512]`.
- [x] `calcSimilarity` returns `double` in `[0.0, 1.0]`.
- [x] `calcAverage` returns element-wise mean.
- [x] Unit tests pass.
- [x] Integration tests pass with sample faces.
- [x] README contains working quickstart.
- [x] Javadoc builds.
- [x] No Python required at runtime.

---

## 9. OpenCode Execution Notes

- Treat this file as the authoritative spec.
- Implement one phase per commit.
- After each phase, run `mvn test`.
- Do not add public API without updating this spec.
- Keep DJL types internal.
- Use `Objects.requireNonNull` for public inputs.
- Wrap all DJL exceptions in `FaceAIException`.
- Model URLs point to the DJL test-models repository (RetinaFace and FaceNet TorchScript exports). Custom URLs may be provided via `FaceAIConfig`. If a URL becomes unavailable, swap in another host and update `ModelConstants` and this document.
- The recognition model must be a TorchScript export from `facenet-pytorch`. The library must not invoke Python.
- Prefer clarity over cleverness. This is a small library.
- **Remaining cleanup (optional, non-blocking):** remove the unused `FaceDetectionTranslator` class once the anchor-based path is confirmed unnecessary.
- **Release notes:** version `0.1.0` (2026-09-11) in `CHANGELOG.md`; MIT license in `LICENSE`.