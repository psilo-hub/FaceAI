package free.svoss.tools.faceai.internal;

import free.svoss.tools.faceai.FaceAIConfig;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.File;

import static org.junit.jupiter.api.Assertions.*;

class ModelLoaderTest {

    @TempDir
    File tempDir;

    // ── ModelConstants ──────────────────────────────────────────────────

    @Test
    void modelConstantsAreDefined() {
        assertNotNull(ModelConstants.RETINAFACE_MODEL_URL);
        assertNotNull(ModelConstants.FACENET_MODEL_URL);
        assertNotNull(ModelConstants.RETINAFACE_MODEL_NAME);
        assertNotNull(ModelConstants.FACENET_MODEL_NAME);
    }

    @Test
    void pytorchEngineNameIsCorrect() {
        assertEquals("PyTorch", ModelConstants.PYTORCH_ENGINE);
    }

    @Test
    void embeddingDimensionIs512() {
        assertEquals(512, ModelConstants.DEFAULT_EMBEDDING_DIMENSION);
    }

    @Test
    void facenetInputSizeIs160() {
        assertEquals(160, ModelConstants.FACENET_INPUT_SIZE);
    }

    @Test
    void retinafaceInputSizeIs640() {
        assertEquals(640, ModelConstants.RETINAFACE_INPUT_SIZE);
    }

    @Test
    void facenetMeanAndStdAreHalves() {
        assertEquals(3, ModelConstants.FACENET_MEAN.length);
        assertEquals(3, ModelConstants.FACENET_STD.length);
        for (float v : ModelConstants.FACENET_MEAN) {
            assertEquals(0.5f, v);
        }
        for (float v : ModelConstants.FACENET_STD) {
            assertEquals(0.5f, v);
        }
    }

    @Test
    void modelUrlsPointToDjlResources() {
        assertTrue(ModelConstants.RETINAFACE_MODEL_URL.startsWith("https://"),
                "RetinaFace URL should be HTTPS");
        assertTrue(ModelConstants.FACENET_MODEL_URL.startsWith("https://"),
                "FaceNet URL should be HTTPS");
    }

    // ── ModelLoader (static utility) ────────────────────────────────────

    @Test
    void createPredictorRequiresNonNullModel() {
        assertThrows(NullPointerException.class,
                () -> ModelLoader.createPredictor(null));
    }

    @Test
    void loadModelRequiresNonNullCriteria() {
        assertThrows(NullPointerException.class,
                () -> ModelLoader.loadModel(null, "TestModel"));
    }

    @Test
    void loadModelRequiresNonNullModelName() {
        assertThrows(NullPointerException.class,
                () -> ModelLoader.loadModel(null, null));
    }

    @Test
    void resolvedCacheDirHonorsExplicitConfig() {
        FaceAIConfig config = FaceAIConfig.builder()
                .cacheDir(tempDir.getAbsolutePath())
                .build();
        File resolved = config.resolvedCacheDir();
        assertEquals(tempDir.getAbsolutePath(), resolved.getAbsolutePath());
    }

    @Test
    void resolvedCacheDirFallsBackToDjlDefault() {
        FaceAIConfig config = FaceAIConfig.builder().build();
        File resolved = config.resolvedCacheDir();
        assertNotNull(resolved);
        assertTrue(resolved.getAbsolutePath().contains(".djl.ai"),
                "Default cache dir should be under ~/.djl.ai");
    }
}