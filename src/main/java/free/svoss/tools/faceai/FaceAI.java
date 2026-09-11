package free.svoss.tools.faceai;

import free.svoss.tools.faceai.internal.EmbeddingUtils;
import free.svoss.tools.faceai.internal.FaceNetRecognizer;
import free.svoss.tools.faceai.internal.RetinaFaceDetector;

import java.awt.image.BufferedImage;
import java.util.List;
import java.util.Objects;

/**
 * Main entry point for face detection and recognition.
 * <p>
 * Models are loaded lazily on first use. The first call to
 * {@link #detectFaces} or {@link #getEmbedding} may be slow due to model
 * download. Not thread-safe by default — each thread should use its own
 * instance or apply external synchronization.
 * <p>
 * Implements {@link AutoCloseable} to release model resources.
 *
 * <pre>{@code
 * try (FaceAI faceai = FaceAI.create()) {
 *     DetectedFace[] faces = faceai.detectFaces(image);
 *     float[] embedding = faceai.getEmbedding(faces[0].crop(image));
 *     double similarity = faceai.calcSimilarity(embedding, otherEmbedding);
 * }
 * }</pre>
 */
public final class FaceAI implements AutoCloseable {

    private final FaceAIConfig config;
    private volatile boolean closed = false;

    // Lazy-loaded models (volatile for safe publication)
    private volatile RetinaFaceDetector detector;
    private volatile FaceNetRecognizer recognizer;

    private FaceAI(FaceAIConfig config) {
        this.config = config;
    }

    /**
     * Creates a FaceAI instance with default configuration.
     *
     * @return a new FaceAI instance
     */
    public static FaceAI create() {
        return new FaceAI(FaceAIConfig.createDefault());
    }

    /**
     * Creates a FaceAI instance with the given configuration.
     *
     * @param config configuration options
     * @return a new FaceAI instance
     * @throws NullPointerException if config is null
     */
    public static FaceAI create(FaceAIConfig config) {
        Objects.requireNonNull(config, "config must not be null");
        return new FaceAI(config);
    }

    /**
     * Detects faces in the given image.
     * <p>
     * The RetinaFace model is lazy-loaded on the first call.
     *
     * @param image the input image (non-null, positive dimensions)
     * @return array of detected faces, sorted by descending confidence; never null
     * @throws NullPointerException if image is null
     * @throws IllegalArgumentException if image has zero dimensions
     * @throws FaceAIException if detection fails
     */
    public DetectedFace[] detectFaces(BufferedImage image) {
        checkNotClosed();
        Objects.requireNonNull(image, "image must not be null");
        if (image.getWidth() <= 0 || image.getHeight() <= 0) {
            throw new IllegalArgumentException("Image must have positive dimensions");
        }
        return detector().detect(image);
    }

    /**
     * Computes a face embedding from a cropped face image.
     * <p>
     * Assumes the image contains exactly one face. Does not run detection.
     * Resizes to 160x160, applies FaceNet preprocessing, returns a 512-dimensional vector.
     * The FaceNet model is lazy-loaded on the first call.
     *
     * @param faceImage a cropped face image (non-null, positive dimensions)
     * @return embedding vector of length 512
     * @throws NullPointerException if faceImage is null
     * @throws FaceAIException if embedding computation fails
     */
    public float[] getEmbedding(BufferedImage faceImage) {
        checkNotClosed();
        Objects.requireNonNull(faceImage, "faceImage must not be null");
        if (faceImage.getWidth() <= 0 || faceImage.getHeight() <= 0) {
            throw new IllegalArgumentException("faceImage must have positive dimensions");
        }
        return recognizer().getEmbedding(faceImage);
    }

    /**
     * Computes cosine-based similarity between two embeddings, mapped to [0, 1].
     *
     * @param embeddingA first embedding (non-null, non-zero norm)
     * @param embeddingB second embedding (non-null, non-zero norm)
     * @return similarity between 0.0 (opposite) and 1.0 (identical direction)
     * @throws IllegalArgumentException if inputs are invalid
     */
    public double calcSimilarity(float[] embeddingA, float[] embeddingB) {
        checkNotClosed();
        return EmbeddingUtils.calcSimilarity(embeddingA, embeddingB);
    }

    /**
     * Computes element-wise arithmetic mean of a list of embeddings.
     *
     * @param embeddings non-null, non-empty list of equal-length embeddings
     * @return average embedding (not L2-normalized)
     * @throws NullPointerException if embeddings is null
     * @throws IllegalArgumentException if list is empty, contains null, or dimensions differ
     */
    public float[] calcAverage(List<float[]> embeddings) {
        checkNotClosed();
        return EmbeddingUtils.calcAverage(embeddings);
    }

    /**
     * Releases model resources.
     */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        if (detector != null) {
            detector.close();
            detector = null;
        }
        if (recognizer != null) {
            recognizer.close();
            recognizer = null;
        }
    }

    private void checkNotClosed() {
        if (closed) {
            throw new FaceAIException("FaceAI instance has been closed");
        }
    }

    /**
     * Lazily loads the RetinaFace detector on first use.
     *
     * @return the detector instance
     * @throws FaceAIException if model loading fails
     */
    private RetinaFaceDetector detector() {
        RetinaFaceDetector d = detector;
        if (d == null) {
            synchronized (this) {
                d = detector;
                if (d == null) {
                    d = new RetinaFaceDetector(config);
                    detector = d;
                }
            }
        }
        return d;
    }

    /**
     * Lazily loads the FaceNet recognizer on first use.
     *
     * @return the recognizer instance
     * @throws FaceAIException if model loading fails
     */
    private FaceNetRecognizer recognizer() {
        FaceNetRecognizer r = recognizer;
        if (r == null) {
            synchronized (this) {
                r = recognizer;
                if (r == null) {
                    r = new FaceNetRecognizer(config);
                    recognizer = r;
                }
            }
        }
        return r;
    }
}
