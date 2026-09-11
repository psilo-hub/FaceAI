package free.svoss.tools.faceai.internal;

/**
 * Model URLs and constants for FaceAI.
 * <p>
 * Models are auto-downloaded on first use and cached locally.
 * URLs point to TorchScript exports suitable for DJL PyTorch engine.
 * When no custom URL is provided via {@code FaceAIConfig}, these defaults are used.
 */
public final class ModelConstants {

    private ModelConstants() {
        // utility class — no instantiation
    }

    // ── Default model URLs ──────────────────────────────────────────────

    /**
     * RetinaFace detection model (TorchScript export).
     * <p>
     * Source: DJL test models repository.
     */
    public static final String RETINAFACE_MODEL_URL =
            "https://resources.djl.ai/test-models/pytorch/retinaface.zip";

    /**
     * FaceNet face feature extraction model (InceptionResnetV1, vggface2).
     * <p>
     * Source: DJL test models repository.
     * Input: 1×3×160×160 RGB. Output: 512-d embedding.
     */
    public static final String FACENET_MODEL_URL =
            "https://resources.djl.ai/test-models/pytorch/face_feature.zip";

    // ── Model names (file prefixes within archives) ─────────────────────

    /** Model file prefix for RetinaFace archive. */
    public static final String RETINAFACE_MODEL_NAME = "retinaface";

    /** Model file prefix for FaceNet archive. */
    public static final String FACENET_MODEL_NAME = "face_feature";

    // ── Engine ──────────────────────────────────────────────────────────

    /** DJL engine name for PyTorch models. */
    public static final String PYTORCH_ENGINE = "PyTorch";

    // ── FaceNet preprocessing constants ─────────────────────────────────

    /** Default embedding dimension for FaceNet output. */
    public static final int DEFAULT_EMBEDDING_DIMENSION = 512;

    /** FaceNet input image size (width and height in pixels). */
    public static final int FACENET_INPUT_SIZE = 160;

    /**
     * FaceNet preprocessing: channel-wise mean values (RGB order).
     * <p>
     * Matches {@code facenet-pytorch} normalization: {@code (pixel / 255 - 0.5) / 0.5}.
     */
    public static final float[] FACENET_MEAN = {0.5f, 0.5f, 0.5f};

    /**
     * FaceNet preprocessing: channel-wise standard deviation (RGB order).
     * <p>
     * Matches {@code facenet-pytorch} normalization: {@code (pixel / 255 - 0.5) / 0.5}.
     */
    public static final float[] FACENET_STD = {0.5f, 0.5f, 0.5f};

    // ── RetinaFace preprocessing constants ──────────────────────────────

    /** RetinaFace default input size for letterbox resize. */
    public static final int RETINAFACE_INPUT_SIZE = 640;
}
