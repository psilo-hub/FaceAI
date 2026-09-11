# FaceAI — Java Face Detection & Recognition Library

## 0. How to Use This Document

This file is the source of truth for OpenCode. Implement the phases in order. After each phase, run the project tests and update the checklist. Do not start optional Phase 9 until Phases 0–8 are complete. Do not add public API beyond what is specified here without updating this document first.

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

Package root: `com.example.faceai`

Internal implementation package: `com.example.faceai.internal`

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
- Prefer DJL Model Zoo artifact if available.
- If no suitable DJL artifact exists, load a TorchScript RetinaFace model from a configured URL.
- Expected output: bounding boxes, confidence scores, and optionally landmarks.
- Required post-processing:
  - Confidence threshold.
  - Box decoding if required by the model format.
  - Non-maximum suppression with IoU threshold `0.4`.
  - Clamp boxes to image bounds.
  - Sort by confidence descending.
- Input size should match the model. Default assumption: letterbox resize to `640x640`, then map boxes back to original coordinates. Document the actual expected size in `RetinaFaceDetector`.

### 4.2 Recognition: FaceNet via facenet-pytorch

- Use FaceNet exported from `facenet-pytorch`.
- The model must be exported to TorchScript. The Java library must not run Python.
- Recommended base model: `InceptionResnetV1(pretrained='vggface2').eval()`.
- Input: `1x3x160x160` RGB tensor.
- Output: `1x512` embedding.
- Preprocessing must match `facenet-pytorch` exactly. Document mean/std and scaling in `FaceNetRecognizer`.
- L2-normalize the embedding before returning if configured.

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

Internal classes:

- `RetinaFaceDetector`
  - Owns detection `Predictor`.
  - Handles preprocessing, inference, and post-processing.
- `FaceNetRecognizer`
  - Owns recognition `Predictor`.
  - Handles preprocessing, inference, and L2 normalization.
- `RetinaFaceTranslator`
  - Converts `BufferedImage` to `NDArray` and model output to `DetectedFace[]`.
- `FaceNetTranslator`
  - Converts `BufferedImage` to `NDArray` and model output to `float[]`.
- `ImageUtils`
  - RGB conversion, alpha compositing over white, resizing, letterboxing.
- `EmbeddingUtils`
  - Cosine similarity, mapping to `[0,1]`, arithmetic mean.
- `ModelLoader`
  - Builds DJL `Criteria`, handles cache dir, wraps exceptions.

Public API must not expose DJL types.

`FaceAI` should lazy-load models on first use and close predictors in `close()`.

Thread safety:

- `FaceAI` is not thread-safe by default.
- Document that each thread should use its own instance or external synchronization.
- Optional Phase 9: thread-local predictors or synchronized wrappers.

---

## 6. Implementation Phases

### Phase 0 — Project Skeleton

- [ ] Create Maven project with Java 11.
- [ ] Add dependencies:
  - [ ] `ai.djl:api`
  - [ ] `ai.djl.pytorch:pytorch-engine`
  - [ ] `ai.djl.pytorch:pytorch-native-auto` with runtime scope
  - [ ] `org.slf4j:slf4j-api`
  - [ ] `org.slf4j:slf4j-simple` for tests
  - [ ] JUnit 5
- [ ] Create packages:
  - [ ] `com.example.faceai`
  - [ ] `com.example.faceai.internal`
- [ ] Add `.gitignore`.
- [ ] Add `README.md` skeleton.
- [ ] Add a placeholder test.

Deliverable: `mvn test` passes with no model loading.

Acceptance: project compiles, tests run, dependency tree is valid.

---

### Phase 1 — Public API and Data Types

- [ ] Implement `DetectedFace`.
- [ ] Implement `FaceAIConfig` with builder and defaults.
- [ ] Implement `FaceAIException`.
- [ ] Create `FaceAI` skeleton with method signatures and Javadoc.
- [ ] Add unit tests for `DetectedFace`:
  - [ ] `bounds()`
  - [ ] `crop()`
  - [ ] `equals` / `hashCode`
- [ ] Add unit tests for `FaceAIConfig` defaults.

Deliverable: public API compiles.

Acceptance: no DJL types appear in public signatures. Unit tests pass.

---

### Phase 2 — Model Loading and Auto-Download

- [ ] Implement `ModelLoader`.
- [ ] Define model constants for RetinaFace and FaceNet.
- [ ] Use DJL `Criteria`.
- [ ] Support `FaceAIConfig.cacheDir`.
- [ ] Respect `DJL_CACHE_DIR`.
- [ ] Log download URL and cache path.
- [ ] Add an integration test tagged `integration` that loads both models.
- [ ] Ensure second run works offline from cache.

Deliverable: models load without manual download.

Acceptance: first run downloads; second run uses cache. Failures throw `FaceAIException` with clear cause.

---

### Phase 3 — Face Detection with RetinaFace

- [ ] Implement `RetinaFaceDetector`.
- [ ] Implement `RetinaFaceTranslator`.
- [ ] Implement RGB conversion.
- [ ] Implement resize/letterbox to model input size.
- [ ] Run inference.
- [ ] Apply confidence threshold.
- [ ] Decode boxes if required by model output.
- [ ] Apply NMS with IoU threshold `0.4`.
- [ ] Clamp boxes to image bounds.
- [ ] Sort detections by confidence descending.
- [ ] Add integration test with a sample face image.
- [ ] Assert at least one face is detected.
- [ ] Assert confidence is above threshold.
- [ ] Assert bounds are inside the image.

Deliverable: `detectFaces` works on real images.

Acceptance: integration test passes. Empty image returns empty array, not `null`.

---

### Phase 4 — Face Recognition Embedding with FaceNet

- [ ] Implement `FaceNetRecognizer`.
- [ ] Implement `FaceNetTranslator`.
- [ ] Implement preprocessing:
  - [ ] RGB conversion.
  - [ ] Resize to `160x160`.
  - [ ] Match `facenet-pytorch` normalization exactly.
- [ ] Run inference.
- [ ] Extract `float[512]`.
- [ ] L2-normalize if configured.
- [ ] Add integration test:
  - [ ] Detect a face.
  - [ ] Crop it.
  - [ ] Get embedding.
  - [ ] Assert length is `512`.
  - [ ] Assert norm is approximately `1.0` if normalization enabled.
  - [ ] Assert same identity similarity is higher than different identity similarity.

Deliverable: `getEmbedding` works on face crops.

Acceptance: integration test passes. Preprocessing is documented in Javadoc.

---

### Phase 5 — Similarity and Average

- [ ] Implement `EmbeddingUtils`.
- [ ] Implement `calcSimilarity`:
  - [ ] Validate inputs.
  - [ ] Normalize internally.
  - [ ] Compute cosine.
  - [ ] Map to `[0,1]`.
  - [ ] Clamp.
- [ ] Implement `calcAverage`:
  - [ ] Validate list.
  - [ ] Validate equal dimensions.
  - [ ] Compute element-wise mean.
- [ ] Unit tests:
  - [ ] Identical vectors return `1.0`.
  - [ ] Orthogonal vectors return `0.5`.
  - [ ] Opposite vectors return `0.0`.
  - [ ] Zero vector throws.
  - [ ] Dimension mismatch throws.
  - [ ] Average of `[1,0]` and `[0,1]` returns `[0.5,0.5]`.
  - [ ] Empty list throws.

Deliverable: utility methods fully tested.

Acceptance: all unit tests pass.

---

### Phase 6 — Facade Integration

- [ ] Implement `FaceAI.create()` and `FaceAI.create(FaceAIConfig)`.
- [ ] Lazy-load models on first use.
- [ ] Wire `detectFaces` to `RetinaFaceDetector`.
- [ ] Wire `getEmbedding` to `FaceNetRecognizer`.
- [ ] Wire `calcSimilarity` and `calcAverage` to `EmbeddingUtils`.
- [ ] Implement `close()`.
- [ ] Add input validation with clear exceptions.
- [ ] Add end-to-end integration test:
  - [ ] Load image.
  - [ ] Detect faces.
  - [ ] Crop first face.
  - [ ] Get embedding.
  - [ ] Compare with another embedding.
  - [ ] Compute average of two embeddings.

Deliverable: all four public methods work end-to-end.

Acceptance: end-to-end test passes. No DJL types leak to public API.

---

### Phase 7 — Testing and Validation

- [ ] Add sample images under `src/test/resources/faces/`:
  - [ ] `person1_a.jpg`
  - [ ] `person1_b.jpg`
  - [ ] `person2.jpg`
- [ ] Add integration test profile in Maven.
- [ ] Tag model-dependent tests with `@Tag("integration")`.
- [ ] Add edge-case tests:
  - [ ] Null image.
  - [ ] Empty image.
  - [ ] Null embeddings.
  - [ ] Wrong embedding dimension.
  - [ ] Zero vector.
  - [ ] Empty list for average.
- [ ] Add model download/cache test.
- [ ] Ensure unit tests run without network.

Deliverable: robust test suite.

Acceptance: `mvn test` runs unit tests. `mvn verify -Pintegration` runs integration tests.

---

### Phase 8 — Documentation and Release

- [ ] Write `README.md`:
  - [ ] Installation.
  - [ ] Quickstart.
  - [ ] Model download and cache behavior.
  - [ ] API examples.
  - [ ] Limitations.
  - [ ] Thread-safety note.
- [ ] Add Javadoc for all public types and methods.
- [ ] Add `CHANGELOG.md`.
- [ ] Set version to `0.1.0`.
- [ ] Add license file if needed.

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

- Unit tests:
  - `DetectedFace`
  - `FaceAIConfig`
  - `EmbeddingUtils`
- Integration tests:
  - Model loading and caching.
  - RetinaFace detection.
  - FaceNet embedding.
  - End-to-end detect → crop → embed → compare.
- Test images:
  - At least two images of the same person.
  - At least one image of a different person.
- Assertions:
  - Same-person similarity > different-person similarity.
  - Detection confidence > threshold.
  - Bounding boxes within image bounds.
  - Embedding length is `512`.
  - Embedding norm is approximately `1.0` when normalization is enabled.
- Integration tests may be slow on first run due to model download. Tag them and document.

---

## 8. Whole-Project Acceptance Criteria

- [ ] All four public methods implemented.
- [ ] Models auto-download on first use.
- [ ] Subsequent runs work from cache.
- [ ] Public API has no DJL types.
- [ ] `detectFaces` returns empty array when no face is found.
- [ ] `getEmbedding` returns `float[512]`.
- [ ] `calcSimilarity` returns `double` in `[0.0, 1.0]`.
- [ ] `calcAverage` returns element-wise mean.
- [ ] Unit tests pass.
- [ ] Integration tests pass with sample faces.
- [ ] README contains working quickstart.
- [ ] Javadoc builds.
- [ ] No Python required at runtime.

---

## 9. OpenCode Execution Notes

- Treat this file as the authoritative spec.
- Implement one phase per commit.
- After each phase, run `mvn test`.
- Do not add public API without updating this spec.
- Keep DJL types internal.
- Use `Objects.requireNonNull` for public inputs.
- Wrap all DJL exceptions in `FaceAIException`.
- If a model URL is unavailable, use the DJL Model Zoo artifact and document the exact artifact name.
- If RetinaFace or FaceNet artifacts are not available in the DJL Model Zoo, define placeholder URLs in `FaceAIConfig` and document how to host the TorchScript files.
- The recognition model must be a TorchScript export from `facenet-pytorch`. The library must not invoke Python.
- Prefer clarity over cleverness. This is a small library.