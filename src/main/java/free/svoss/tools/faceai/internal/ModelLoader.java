package free.svoss.tools.faceai.internal;

import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import free.svoss.tools.faceai.FaceAIException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Objects;

/**
 * Internal utility for loading DJL models via {@link Criteria}.
 * <p>
 * Handles cache directory setup, model download logging, and exception wrapping.
 * All DJL-specific exceptions are converted to {@link FaceAIException}.
 *
 * <p><strong>This class is internal — not part of the public API.</strong>
 */
public final class ModelLoader {

    private static final Logger logger = LoggerFactory.getLogger(ModelLoader.class);

    private ModelLoader() {
        // utility class — no instantiation
    }

    /**
     * Loads a DJL {@link ZooModel} from the given criteria using DJL's default
     * cache resolution ({@code DJL_CACHE_DIR} env var, then {@code DJL_CACHE_DIR}
     * system property, then {@code ~/.djl.ai}).
     *
     * @param <I>       model input type
     * @param <O>       model output type
     * @param criteria  the DJL criteria describing the model to load
     * @param modelName logical name for logging (e.g. "RetinaFace", "FaceNet")
     * @return a new {@link ZooModel} instance (caller must close)
     * @throws FaceAIException if model loading fails
     */
    public static <I, O> ZooModel<I, O> loadModel(
            Criteria<I, O> criteria, String modelName) {
        return loadModel(criteria, modelName, null);
    }

    /**
     * Loads a DJL {@link ZooModel} from the given criteria.
     * <p>
     * The model is downloaded on first use and cached by DJL. Subsequent calls
     * use the cached version. This method logs the download URL and cache path.
     * <p>
     * When {@code cacheDir} is non-null and non-empty, it is honored by setting
     * the {@code DJL_CACHE_DIR} system property before loading. DJL 0.31.0
     * resolves the cache directory via {@code ai.djl.util.Utils#getCacheDir()},
     * which checks the {@code DJL_CACHE_DIR} environment variable first, then the
     * {@code DJL_CACHE_DIR} system property, then defaults to {@code ~/.djl.ai}
     * (verified from the api-0.31.0.jar bytecode). All DJL-specific exceptions are
     * converted to {@link FaceAIException}.
     *
     * @param <I>       model input type
     * @param <O>       model output type
     * @param criteria  the DJL criteria describing the model to load
     * @param modelName logical name for logging (e.g. "RetinaFace", "FaceNet")
     * @param cacheDir  cache directory to use, or {@code null} for DJL default
     *                  resolution (DJL_CACHE_DIR env/property, then ~/.djl.ai)
     * @return a new {@link ZooModel} instance (caller must close)
     * @throws FaceAIException if model loading fails
     */
    public static <I, O> ZooModel<I, O> loadModel(
            Criteria<I, O> criteria, String modelName, String cacheDir) {

        Objects.requireNonNull(criteria, "criteria must not be null");
        Objects.requireNonNull(modelName, "modelName must not be null");

        if (cacheDir != null && !cacheDir.isEmpty()) {
            String resolvedCache = new File(cacheDir).getAbsolutePath();
            // DJL reads the DJL_CACHE_DIR env var first, then the same-named
            // system property (Utils.getEnvOrSystemProperty). We set the property.
            System.setProperty("DJL_CACHE_DIR", resolvedCache);
            logger.info("Model cache directory: {}", resolvedCache);
        } else {
            String env = System.getenv("DJL_CACHE_DIR");
            String prop = System.getProperty("DJL_CACHE_DIR");
            String resolved = (env != null && !env.isEmpty())
                    ? env
                    : ((prop != null && !prop.isEmpty()) ? prop : null);
            logger.info("Model cache directory: {}",
                    resolved != null ? resolved : "~/.djl.ai (DJL default)");
        }

        String modelUrl = defaultUrlFor(modelName);
        logger.info("Loading {} model from {}", modelName,
                modelUrl != null ? modelUrl : "(criteria-defined URL — see detector logs)");
        logger.debug("Criteria: {}", criteria);

        try {
            ZooModel<I, O> model = criteria.loadModel();
            logger.info("Successfully loaded {} model", modelName);
            return model;
        } catch (ModelNotFoundException e) {
            throw new FaceAIException(
                    "Model not found for " + modelName
                    + ". Ensure the model URL is accessible and the archive "
                    + "contains the expected model files.", e);
        } catch (MalformedModelException e) {
            throw new FaceAIException(
                    "Malformed model data for " + modelName + ": " + e.getMessage(), e);
        } catch (IOException e) {
            throw new FaceAIException(
                    "Failed to load " + modelName
                    + " model — check your network connection and cache directory.", e);
        } catch (Exception e) {
            throw new FaceAIException(
                    "Unexpected error loading " + modelName + " model: " + e.getMessage(), e);
        }
    }

    /**
     * Returns the default model URL for a well-known model name, or {@code null}.
     * Used for logging only — {@code Criteria} does not expose its model URLs
     * publicly in DJL 0.31.0 (verified via javap).
     */
    private static String defaultUrlFor(String modelName) {
        if ("RetinaFace".equals(modelName)) {
            return ModelConstants.RETINAFACE_MODEL_URL;
        }
        if ("FaceNet".equals(modelName)) {
            return ModelConstants.FACENET_MODEL_URL;
        }
        return null;
    }

    /**
     * Creates a {@link Predictor} from a loaded {@link ZooModel}.
     *
     * @param <I>   input type
     * @param <O>   output type
     * @param model the loaded model
     * @return a new predictor
     * @throws FaceAIException if predictor creation fails
     */
    public static <I, O> Predictor<I, O> createPredictor(ZooModel<I, O> model) {
        Objects.requireNonNull(model, "model must not be null");
        try {
            return model.newPredictor();
        } catch (Exception e) {
            throw new FaceAIException(
                    "Failed to create predictor: " + e.getMessage(), e);
        }
    }
}