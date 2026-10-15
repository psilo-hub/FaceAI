# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-10-15

### Added
- `DetectedFace` now exposes the RetinaFace 5-point landmarks (left eye, right eye, nose tip, left mouth corner, right mouth corner) via `hasLandmarks()` and `landmarks()`, as a source- and binary-compatible API addition (the existing 5-argument constructor is preserved).
- `DetectedFace.rollAngleDegrees()` returns the in-plane face rotation (roll) derived from the eye line: `0` = upright, positive = clockwise, normalised to `[-90, 90]`; `NaN` when landmarks are unavailable.
- `RetinaFaceTranslator` decodes the `landms` output tensor and attaches the landmarks to each detection (guarded when the model export does not return the third tensor).
- Legacy `FaceDetectionTranslator` now passes its decoded landmarks through to `DetectedFace`.
- Unit tests for the landmark API and the `decodeLandmarks` helper, plus an integration test asserting landmarks fall within the bounding box.

## [0.1.0] - 2026-09-11

### Added
- Face detection using RetinaFace (DJL PyTorch engine)
- Face recognition using FaceNet (InceptionResnetV1, vggface2)
- Automatic model download and local caching
- Cosine-based similarity mapping to [0.0, 1.0]
- Arithmetic average of embeddings
- FaceAIConfig builder with configurable cache directory, thresholds, and embedding dimension
- Unit tests for all core components (82 tests)
- Integration test suite for end-to-end pipeline validation
