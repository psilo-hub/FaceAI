package free.svoss.tools.faceai.internal;

import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;

import free.svoss.tools.faceai.FaceAIConfig;
import free.svoss.tools.faceai.FaceAIException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.image.BufferedImage;
import java.util.Objects;

/**
 * Computes face embeddings using a FaceNet model via DJL.
 * <p>
 * Handles model lifecycle and wraps exceptions. Owns a
 * {@link Predictor} for inference. Returns a
 * {@value ModelConstants#DEFAULT_EMBEDDING_DIMENSION}-dimensional
 * float array for each face image.
 *
 * <p><strong>This class is internal — not part of the public API.</strong>
 */
public final class FaceNetRecognizer implements AutoCloseable {

    private static final Logger logger = LoggerFactory.getLogger(FaceNetRecognizer.class);

    private final Predictor<BufferedImage, float[]> predictor;
    private final FaceAIConfig config;
    private ZooModel<BufferedImage, float[]> model;

    /**
     * Creates a recognizer with an externally-provided predictor (for testing).
     *
     * @param predictor the predictor to use
     * @param config    the FaceAI config
     */
    public FaceNetRecognizer(Predictor<BufferedImage, float[]> predictor, FaceAIConfig config) {
        this.predictor = Objects.requireNonNull(predictor, "predictor must not be null");
        this.config = Objects.requireNonNull(config, "config must not be null");
    }

    /**
     * Creates a recognizer by loading the FaceNet model.
     *
     * @param config the FaceAI config
     */
    public FaceNetRecognizer(FaceAIConfig config) {
        Objects.requireNonNull(config, "config must not be null");
        this.config = config;

        FaceNetTranslator translator = new FaceNetTranslator(config);

        String modelUrl = config.recognitionModelUrl() != null
                ? config.recognitionModelUrl()
                : ModelConstants.FACENET_MODEL_URL;
        String modelName = ModelConstants.FACENET_MODEL_NAME;

        Criteria<BufferedImage, float[]> criteria = Criteria.builder()
                .setTypes(BufferedImage.class, float[].class)
                .optModelUrls(modelUrl)
                .optModelName(modelName)
                .optEngine(ModelConstants.PYTORCH_ENGINE)
                .optTranslator(translator)
                .build();

        this.model = ModelLoader.loadModel(criteria, "FaceNet",
                config.resolvedCacheDir().getAbsolutePath());
        this.predictor = ModelLoader.createPredictor(model);
        logger.info("FaceNet recognizer initialized");
    }

    /**
     * Computes a face embedding from a cropped face image.
     *
     * @param faceImage a cropped face image (non-null, positive dimensions)
     * @return embedding vector of length {@code config.embeddingDimension()}
     * @throws FaceAIException if embedding computation fails
     */
    public float[] getEmbedding(BufferedImage faceImage) {
        Objects.requireNonNull(faceImage, "faceImage must not be null");
        if (faceImage.getWidth() <= 0 || faceImage.getHeight() <= 0) {
            throw new IllegalArgumentException("faceImage must have positive dimensions");
        }
        try {
            float[] embedding = predictor.predict(faceImage);
            if (embedding == null) {
                throw new FaceAIException("Model returned null embedding");
            }
            if (embedding.length != config.embeddingDimension()) {
                throw new FaceAIException(
                        "Expected embedding dimension " + config.embeddingDimension()
                        + " but got " + embedding.length);
            }
            return embedding;
        } catch (FaceAIException e) {
            throw e;
        } catch (Exception e) {
            throw new FaceAIException("Face embedding computation failed: " + e.getMessage(), e);
        }
    }

    @Override
    public void close() {
        if (predictor != null) {
            predictor.close();
        }
        if (model != null) {
            model.close();
        }
    }
}