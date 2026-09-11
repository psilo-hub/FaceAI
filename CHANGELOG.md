# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
