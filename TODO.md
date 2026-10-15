# TODO — Face rotation (roll) detection

Derived from `REQUIRED_CHANGES_FACE_ROTATION.md`. Checked off as each item is
implemented and committed.

## 1. Public API — `DetectedFace` (§3.1)
- [x] Add `float[] landmarks` field (10 floats or `null`).
- [x] Keep the 5-arg constructor; add the 6-arg constructor that validates length 10 and clones.
- [x] Add `hasLandmarks()`, `landmarks()` (defensive copy), `rollAngleDegrees()`.
- [x] Include landmarks in `equals` / `hashCode` / `toString` (null-safe).

## 2. Active detection path — `RetinaFaceTranslator` (§3.2)
- [x] Read `landms` from `list.get(2)` only when `list.size() > 2`.
- [x] Extract package-private `decodeLandmarks(...)` helper (VARIANCE_IN = 0.1, clamped to pixel coords).
- [x] Attach landmarks to the matching `DetectedFace` so they survive filtering/sorting/NMS.

## 3. Legacy path — `FaceDetectionTranslator` (§3.3)
- [x] Pass the already-decoded `landms` through to `DetectedFace`.

## 4. Unit tests (§3.5)
- [x] `DetectedFaceTest`: `hasLandmarks` / `landmarks` defensive copy.
- [x] `DetectedFaceTest`: `rollAngleDegrees()` synthetic eye positions (0, 90, -90, 180→0).
- [x] `DetectedFaceTest`: `Double.NaN` when absent; 5-arg ctor has no landmarks; length != 10 throws.
- [x] `DetectedFaceTest`: `equals` / `hashCode` distinguish different landmarks.
- [x] `RetinaFaceTranslatorTest`: unit test `decodeLandmarks` helper (no model required).

## 5. Integration test (§3.5)
- [ ] `@Tag("integration")` test: real model, `hasLandmarks()`, each landmark within bbox margin.

## 6. Documentation (§3.4)
- [ ] `PROJECT_SPEC.md` §3.2: replace "Optional future field" with the new field/constructor/`rollAngleDegrees()`.
- [ ] `PROJECT_SPEC.md` §4.1: document landmark decoding in the post-processing list.
- [ ] `PROJECT_SPEC.md` Phase 9: tick `5-point landmarks in DetectedFace`.

## 7. Versioning and release (§3.4)
- [ ] Add `CHANGELOG.md` entry for `0.2.0`.
- [ ] Bump version `0.1.0` → `0.2.0` in `pom.xml`.
- [ ] `mvn install` so the consumer can resolve `free.svoss.tools:faceai:0.2.0`.

## 8. Verification
- [ ] `mvn test` green.
- [ ] `mvn verify -Pintegration` green (model download required).

## Not required (optional, left for the consumer — §4)
- [ ] Four-orientation `detectFacesUpright` convenience — consumer implements the search.
- [ ] `alignedCrop(...)` alignment helper — consumer can rotate with Java2D from `rollAngleDegrees()`.
