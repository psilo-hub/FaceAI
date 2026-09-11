package free.svoss.tools.faceai;

import java.io.File;

/**
 * Configuration for FaceAI.
 * Use the builder to customize defaults.
 */
public final class FaceAIConfig {

    private final String cacheDir;
    private final float detectionThreshold;
    private final float nmsThreshold;
    private final int embeddingDimension;
    private final String device;
    private final boolean l2NormalizeEmbeddings;
    private final String detectionModelUrl;
    private final String recognitionModelUrl;

    private FaceAIConfig(Builder builder) {
        this.cacheDir = builder.cacheDir;
        this.detectionThreshold = builder.detectionThreshold;
        this.nmsThreshold = builder.nmsThreshold;
        this.embeddingDimension = builder.embeddingDimension;
        this.device = builder.device;
        this.l2NormalizeEmbeddings = builder.l2NormalizeEmbeddings;
        this.detectionModelUrl = builder.detectionModelUrl;
        this.recognitionModelUrl = builder.recognitionModelUrl;
    }

    /**
     * Returns a configuration with all default values.
     *
     * @return a new default configuration instance
     */
    public static FaceAIConfig createDefault() { return new Builder().build(); }

    /** Returns the explicit cache directory, or {@code null} if not set. */
    public String cacheDir() { return cacheDir; }
    /** Returns the minimum detection confidence threshold (default 0.8). */
    public float detectionThreshold() { return detectionThreshold; }
    /** Returns the NMS IoU threshold (default 0.4). */
    public float nmsThreshold() { return nmsThreshold; }
    /** Returns the embedding dimensionality (default 512). */
    public int embeddingDimension() { return embeddingDimension; }
    /** Returns the compute device identifier (default "CPU"). */
    public String device() { return device; }
    /** Returns whether L2 normalization is applied to embeddings (default true). */
    public boolean l2NormalizeEmbeddings() { return l2NormalizeEmbeddings; }
    /** Returns the custom RetinaFace model URL, or {@code null} if using the default. */
    public String detectionModelUrl() { return detectionModelUrl; }
    /** Returns the custom FaceNet model URL, or {@code null} if using the default. */
    public String recognitionModelUrl() { return recognitionModelUrl; }

    /**
     * Returns the resolved cache directory path.
     * Priority: explicit cacheDir > DJL_CACHE_DIR env > ~/.djl.ai/cache
     */
    public File resolvedCacheDir() {
        if (cacheDir != null && !cacheDir.isEmpty()) {
            return new File(cacheDir);
        }
        String envCache = System.getenv("DJL_CACHE_DIR");
        if (envCache != null && !envCache.isEmpty()) {
            return new File(envCache);
        }
        return new File(System.getProperty("user.home"), ".djl.ai/cache");
    }

    public static Builder builder() { return new Builder(); }

    /**
     * Builder for {@link FaceAIConfig}.
     * <p>
     * All properties have sensible defaults; call only the setters you need.
     */
    public static final class Builder {
        private String cacheDir = "";
        private float detectionThreshold = 0.8f;
        private float nmsThreshold = 0.4f;
        private int embeddingDimension = 512;
        private String device = "CPU";
        private boolean l2NormalizeEmbeddings = true;
        private String detectionModelUrl;
        private String recognitionModelUrl;

        private Builder() {}

        /**
         * Sets the model cache directory. Overrides {@code DJL_CACHE_DIR} and
         * the default {@code ~/.djl.ai/cache}.
         *
         * @param cacheDir the cache directory path
         * @return this builder
         */
        public Builder cacheDir(String cacheDir) { this.cacheDir = cacheDir; return this; }
        /** Sets the minimum detection confidence threshold. Default: 0.8. */
        public Builder detectionThreshold(float detectionThreshold) { this.detectionThreshold = detectionThreshold; return this; }
        /** Sets the NMS IoU threshold. Default: 0.4. */
        public Builder nmsThreshold(float nmsThreshold) { this.nmsThreshold = nmsThreshold; return this; }
        /** Sets the expected FaceNet embedding dimension. Default: 512. */
        public Builder embeddingDimension(int embeddingDimension) { this.embeddingDimension = embeddingDimension; return this; }
        /** Sets the compute device (e.g. "CPU"). Default: "CPU". */
        public Builder device(String device) { this.device = device; return this; }
        /** Sets whether to L2-normalize embeddings. Default: true. */
        public Builder l2NormalizeEmbeddings(boolean l2Normalize) { this.l2NormalizeEmbeddings = l2Normalize; return this; }
        /** Sets a custom RetinaFace model URL. Default: {@code null} (use built-in URL). */
        public Builder detectionModelUrl(String url) { this.detectionModelUrl = url; return this; }
        /** Sets a custom FaceNet model URL. Default: {@code null} (use built-in URL). */
        public Builder recognitionModelUrl(String url) { this.recognitionModelUrl = url; return this; }

        /**
         * Builds the configuration.
         *
         * @return a new {@link FaceAIConfig} instance
         */
        public FaceAIConfig build() { return new FaceAIConfig(this); }
    }
}
