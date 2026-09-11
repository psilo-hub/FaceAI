package free.svoss.tools.faceai.internal;

import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;

import free.svoss.tools.faceai.DetectedFace;
import free.svoss.tools.faceai.FaceAIConfig;
import free.svoss.tools.faceai.FaceAIException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.image.BufferedImage;
import java.util.Objects;

/**
 * Face detection using the RetinaFace model via DJL.
 * <p>
 * Wraps a DJL {@code Predictor<BufferedImage, DetectedFace[]>} and delegates
 * to it for inference. Provides two construction paths:
 * <ul>
 *   <li>inject an existing predictor (for testing), or</li>
 *   <li>pass a {@link FaceAIConfig} to load the model automatically via
 *       {@link ModelLoader} (used by the {@code FaceAI} facade).</li>
 * </ul>
 *
 * <p><strong>This class is internal — not part of the public API.</strong>
 */
public final class RetinaFaceDetector implements AutoCloseable {

    private static final Logger logger = LoggerFactory.getLogger(RetinaFaceDetector.class);

    private final Predictor<BufferedImage, DetectedFace[]> predictor;
    private ZooModel<BufferedImage, DetectedFace[]> model;

    /**
     * Creates a RetinaFaceDetector with an externally-provided predictor
     * (for testing).
     *
     * @param predictor the DJL predictor for RetinaFace inference
     */
    public RetinaFaceDetector(Predictor<BufferedImage, DetectedFace[]> predictor) {
        this.predictor = Objects.requireNonNull(predictor, "predictor must not be null");
    }

    /**
     * Creates a RetinaFaceDetector by loading the RetinaFace model.
     *
     * @param config the FaceAI config
     */
    public RetinaFaceDetector(FaceAIConfig config) {
        Objects.requireNonNull(config, "config must not be null");

        RetinaFaceTranslator translator = new RetinaFaceTranslator(config);

        String modelUrl = config.detectionModelUrl() != null
                ? config.detectionModelUrl()
                : ModelConstants.RETINAFACE_MODEL_URL;
        String modelName = ModelConstants.RETINAFACE_MODEL_NAME;

        Criteria<BufferedImage, DetectedFace[]> criteria = Criteria.builder()
                .setTypes(BufferedImage.class, DetectedFace[].class)
                .optModelUrls(modelUrl)
                .optModelName(modelName)
                .optEngine(ModelConstants.PYTORCH_ENGINE)
                .optTranslator(translator)
                .build();

        this.model = ModelLoader.loadModel(criteria, "RetinaFace",
                config.resolvedCacheDir().getAbsolutePath());
        this.predictor = ModelLoader.createPredictor(model);
        logger.info("RetinaFace detector initialized");
    }

    /**
     * Detects faces in the given image, sorted by descending confidence.
     * Never returns null — empty array if no faces are found.
     *
     * @param image the input image (non-null, positive dimensions)
     * @return array of detected faces
     * @throws NullPointerException if image is null
     * @throws FaceAIException      if detection fails
     */
    public DetectedFace[] detect(BufferedImage image) {
        Objects.requireNonNull(image, "image must not be null");
        logger.debug("Running face detection on {}x{} image", image.getWidth(), image.getHeight());
        try {
            DetectedFace[] faces = predictor.predict(image);
            logger.debug("Detected {} faces", faces.length);
            return faces == null ? new DetectedFace[0] : faces;
        } catch (FaceAIException e) {
            throw e;
        } catch (Exception e) {
            throw new FaceAIException("Face detection failed: " + e.getMessage(), e);
        }
    }

    /**
     * Releases the underlying predictor and, if this instance loaded the
     * model itself, the model as well.
     */
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