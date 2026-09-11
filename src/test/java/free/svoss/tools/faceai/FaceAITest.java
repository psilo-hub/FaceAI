package free.svoss.tools.faceai;

import org.junit.jupiter.api.Test;

import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class FaceAITest {

    @Test
    void createReturnsWorkingInstance() {
        try (FaceAI faceai = FaceAI.create()) {
            assertNotNull(faceai);
        }
    }

    @Test
    void createAcceptsCustomConfig() {
        FaceAIConfig config = FaceAIConfig.builder().build();
        try (FaceAI faceai = FaceAI.create(config)) {
            assertNotNull(faceai);
        }
    }

    @Test
    void createRejectsNullConfig() {
        assertThrows(NullPointerException.class, () -> FaceAI.create(null));
    }

    @Test
    void detectFacesRejectsNullImage() {
        try (FaceAI faceai = FaceAI.create()) {
            assertThrows(NullPointerException.class, () -> faceai.detectFaces(null));
        }
    }

    @Test
    void getEmbeddingRejectsNullFaceImage() {
        try (FaceAI faceai = FaceAI.create()) {
            assertThrows(NullPointerException.class, () -> faceai.getEmbedding(null));
        }
    }

    @Test
    void calcSimilarityDelegatesToEmbeddingUtils() {
        try (FaceAI faceai = FaceAI.create()) {
            float[] a = {1.0f, 0.0f, 0.0f};
            float[] b = {1.0f, 0.0f, 0.0f};
            assertEquals(1.0, faceai.calcSimilarity(a, b), 1e-6);
        }
    }

    @Test
    void calcSimilarityRejectsInvalidInputs() {
        try (FaceAI faceai = FaceAI.create()) {
            assertThrows(IllegalArgumentException.class,
                    () -> faceai.calcSimilarity(new float[3], new float[4]));
            assertThrows(IllegalArgumentException.class,
                    () -> faceai.calcSimilarity(new float[3], new float[3])); // zero vector
            assertThrows(NullPointerException.class,
                    () -> faceai.calcSimilarity(null, new float[3]));
        }
    }

    @Test
    void calcAverageDelegatesToEmbeddingUtils() {
        try (FaceAI faceai = FaceAI.create()) {
            List<float[]> embeddings = Arrays.asList(
                    new float[]{1.0f, 2.0f},
                    new float[]{3.0f, 4.0f});
            float[] avg = faceai.calcAverage(embeddings);
            assertArrayEquals(new float[]{2.0f, 3.0f}, avg, 1e-6f);
        }
    }

    @Test
    void calcAverageRejectsInvalidInputs() {
        try (FaceAI faceai = FaceAI.create()) {
            assertThrows(NullPointerException.class, () -> faceai.calcAverage(null));
            assertThrows(IllegalArgumentException.class,
                    () -> faceai.calcAverage(Arrays.asList()));
            assertThrows(IllegalArgumentException.class,
                    () -> faceai.calcAverage(Arrays.asList(
                            new float[]{1.0f}, new float[]{1.0f, 2.0f})));
        }
    }

    @Test
    void closedInstanceThrowsOnAllOperations() {
        FaceAI faceai = FaceAI.create();
        faceai.close();
        assertThrows(FaceAIException.class, () -> faceai.detectFaces(
                new BufferedImage(10, 10, BufferedImage.TYPE_INT_RGB)));
        assertThrows(FaceAIException.class, () -> faceai.getEmbedding(
                new BufferedImage(10, 10, BufferedImage.TYPE_INT_RGB)));
        assertThrows(FaceAIException.class,
                () -> faceai.calcSimilarity(new float[]{1f}, new float[]{1f}));
        assertThrows(FaceAIException.class, () -> faceai.calcAverage(
                Arrays.asList(new float[]{1f})));
        // close() is idempotent
        faceai.close();
    }
}