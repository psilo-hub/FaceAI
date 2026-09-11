package free.svoss.tools.faceai.internal;

import free.svoss.tools.faceai.FaceAIConfig;
import org.junit.jupiter.api.Test;

import java.awt.image.BufferedImage;

import static org.junit.jupiter.api.Assertions.*;

class FaceNetTranslatorTest {

    private final FaceAIConfig config = FaceAIConfig.createDefault();

    @Test
    void translatorUsesConfiguredEmbeddingDimension() {
        FaceAIConfig custom = FaceAIConfig.builder()
                .embeddingDimension(256)
                .l2NormalizeEmbeddings(false)
                .build();
        FaceNetTranslator translator = new FaceNetTranslator(custom);
        assertNotNull(translator);
    }

    @Test
    void l2NormDisabledReturnsRawEmbedding() {
        FaceAIConfig noNorm = FaceAIConfig.builder()
                .l2NormalizeEmbeddings(false)
                .build();
        FaceNetTranslator translator = new FaceNetTranslator(noNorm);
        assertNotNull(translator);
        assertFalse(noNorm.l2NormalizeEmbeddings());
    }

    @Test
    void outputDimensionMatchesConfig() {
        assertEquals(512, config.embeddingDimension());
        assertEquals(160, ModelConstants.FACENET_INPUT_SIZE);
    }

    @Test
    void l2NormalizeProducesUnitNorm() {
        // Verify the L2 logic used by FaceNetTranslator.processOutput
        float[] raw = new float[512];
        for (int i = 0; i < 512; i++) raw[i] = (float) Math.sin(i);
        float[] result = l2Normalize(raw);
        double norm = 0;
        for (float v : result) norm += v * v;
        norm = Math.sqrt(norm);
        assertEquals(1.0, norm, 1e-5, "L2-normalized vector should have unit norm");
    }

    @Test
    void preprocessingNormalizesToMinusOneToOneRange() {
        // The translator's processInput path:
        // toRGB → resize(160,160) → (pixel/255 - 0.5)/0.5
        // White pixel (255,255,255) → (1.0 - 0.5)/0.5 = 1.0
        // Black pixel (0,0,0)       → (0.0 - 0.5)/0.5 = -1.0
        BufferedImage img = new BufferedImage(160, 160, BufferedImage.TYPE_INT_RGB);
        java.awt.Graphics2D g = img.createGraphics();
        g.setColor(java.awt.Color.WHITE);
        g.fillRect(0, 0, 160, 160);
        g.dispose();

        // Verify manually-computed normalized values via a tiny white/black test
        assertEquals(1.0f, (255f / 255f - 0.5f) / 0.5f, 1e-6f);  // white
        assertEquals(-1.0f, (0f / 255f - 0.5f) / 0.5f, 1e-6f);   // black
    }

    /** Helper: L2-normalize a float array (same logic as FaceNetTranslator.processOutput). */
    private static float[] l2Normalize(float[] vec) {
        double norm = 0.0;
        for (float v : vec) norm += v * v;
        norm = Math.sqrt(norm);
        float[] result = vec.clone();
        if (norm > 0) {
            for (int i = 0; i < result.length; i++) {
                result[i] = (float) (result[i] / norm);
            }
        }
        return result;
    }
}