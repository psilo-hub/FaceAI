# FaceAI — Simple Java Face Detection & Recognition

A lightweight Java library for face detection and face recognition using
[DJL](https://djl.ai) (Deep Java Library) with PyTorch models.

## Features

- **Face detection** — RetinaFace (MobileNet-0.25 backbone) via DJL PyTorch
- **Face recognition** — FaceNet (InceptionResnetV1) 512-dimensional embeddings
- **Cosine similarity** — embedding comparison mapped to [0, 1]
- **Embedding averaging** — combine multiple embeddings into one reference
- **Configurable** — thresholds, cache directory, device, model URLs
- **Auto-download** — models download on first use and are cached locally
- **Clean public API** — no DJL types leak into the public surface

## Requirements

- Java 11+
- Maven 3.6+
- Internet connection on first run (model download ~200 MB total)

## Maven Dependency

Add to your `pom.xml` (after `mvn install`):

```xml
<dependency>
    <groupId>com.example</groupId>
    <artifactId>faceai</artifactId>
    <version>0.1.0</version>
</dependency>
```

## Quick Start

```java
import free.svoss.tools.faceai.DetectedFace;
import free.svoss.tools.faceai.FaceAI;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;

try(FaceAI faceai = FaceAI.create()){
BufferedImage image = ImageIO.read(new File("photo.jpg"));

// 1. Detect faces (sorted by descending confidence)
DetectedFace[] faces = faceai.detectFaces(image);

// 2. Compute an embedding for the first (highest-confidence) face
float[] embedding = faceai.getEmbedding(faces[0].crop(image));

// 3. Compare against a reference embedding (same person = higher score)
double similarity = faceai.calcSimilarity(embedding, referenceEmbedding);
// 1.0 = identical direction, 0.5 = orthogonal, 0.0 = opposite

// 4. Average multiple embeddings (e.g. enrollment) into one reference
float[] average = faceai.calcAverage(java.util.List.of(embedding, otherEmbedding));
}
```

## Configuration

```java
FaceAIConfig config = FaceAIConfig.builder()
        .cacheDir("C:/faceai-models")       // model cache location
        .detectionThreshold(0.7f)           // min confidence (default 0.8)
        .nmsThreshold(0.4f)                 // NMS IoU threshold (default 0.4)
        .embeddingDimension(512)            // FaceNet output size (default 512)
        .device("CPU")                      // compute device (default CPU)
        .l2NormalizeEmbeddings(true)        // L2-normalize embeddings (default true)
        .detectionModelUrl("https://...")   // override RetinaFace model URL
        .recognitionModelUrl("https://...") // override FaceNet model URL
        .build();

try (FaceAI faceai = FaceAI.create(config)) {
    // ...
}
```

If no `cacheDir` is set, models are cached under `$DJL_CACHE_DIR`
(env var or system property) or `~/.djl.ai/cache` by default.

## API Reference

### Detect faces

```java
DetectedFace[] faces = faceai.detectFaces(image);
for (DetectedFace face : faces) {
    System.out.printf("Face at (%d,%d) %dx%d conf=%.3f%n",
            face.x(), face.y(), face.width(), face.height(), face.confidence());
}
```

### Extract an embedding

```java
BufferedImage crop = faces[0].crop(image);
float[] embedding = faceai.getEmbedding(crop);  // 512-dimensional vector
```

### Compare two embeddings

```java
double score = faceai.calcSimilarity(embedding, reference);
// 1.0 = same person, 0.5 = orthogonal, 0.0 = opposite direction
```

### Average embeddings (enrollment)

```java
float[] avg = faceai.calcAverage(List.of(emb1, emb2, emb3));
```

`DetectedFace` exposes `x()`, `y()`, `width()`, `height()`, `confidence()`,
`bounds()` (a `java.awt.Rectangle`) and `crop(BufferedImage)`.

## Building

```bash
mvn clean test                 # unit tests (no model downloads)
mvn verify -Pintegration       # end-to-end tests (downloads models on first run)
```

## Project Structure

```
src/main/java/com/example/faceai/
├── FaceAI.java               # public facade
├── DetectedFace.java         # value object
├── FaceAIConfig.java         # builder-based configuration
├── FaceAIException.java      # runtime exception wrapping DJL errors
└── internal/                 # implementation (not public API)
    ├── ModelLoader.java      # criteria-based loading + caching
    ├── ModelConstants.java   # model URLs and preprocessing constants
    ├── RetinaFaceTranslator.java / RetinaFaceDetector.java
    ├── FaceNetTranslator.java / FaceNetRecognizer.java
    ├── ImageUtils.java       # RGB, resize, letterbox, NDArray conversion
    ├── PostProcessing.java   # thresholding, NMS, box decode, sorting
    └── EmbeddingUtils.java   # similarity + average (pure Java)
```

## License

See project for details.