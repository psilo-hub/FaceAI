package free.svoss.tools.faceai;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class FaceAIConfigTest {

    @Test
    void defaultConfigHasExpectedValues() {
        FaceAIConfig config = FaceAIConfig.createDefault();
        assertEquals(0.8f, config.detectionThreshold());
        assertEquals(0.4f, config.nmsThreshold());
        assertEquals(512, config.embeddingDimension());
        assertEquals("CPU", config.device());
        assertTrue(config.l2NormalizeEmbeddings());
        assertNull(config.detectionModelUrl());
        assertNull(config.recognitionModelUrl());
    }

    @Test
    void builderSetsValues() {
        FaceAIConfig config = FaceAIConfig.builder()
                .detectionThreshold(0.5f)
                .nmsThreshold(0.3f)
                .embeddingDimension(256)
                .device("GPU")
                .l2NormalizeEmbeddings(false)
                .cacheDir("/tmp/cache")
                .detectionModelUrl("http://example.com/detector.pt")
                .recognitionModelUrl("http://example.com/recognizer.pt")
                .build();
        assertEquals(0.5f, config.detectionThreshold());
        assertEquals(0.3f, config.nmsThreshold());
        assertEquals(256, config.embeddingDimension());
        assertEquals("GPU", config.device());
        assertFalse(config.l2NormalizeEmbeddings());
        assertEquals("/tmp/cache", config.cacheDir());
        assertEquals("http://example.com/detector.pt", config.detectionModelUrl());
        assertEquals("http://example.com/recognizer.pt", config.recognitionModelUrl());
    }

    @Test
    void resolvedCacheDirUsesDefault() {
        FaceAIConfig config = FaceAIConfig.createDefault();
        // Should resolve to ~/.djl.ai/cache (or DJL_CACHE_DIR if set)
        assertNotNull(config.resolvedCacheDir());
    }

    @Test
    void resolvedCacheDirUsesExplicitValue() {
        FaceAIConfig config = FaceAIConfig.builder().cacheDir("/my/cache").build();
        // On Windows, File normalizes slashes to backslashes.
        // Compare the tail of the path to stay platform-independent.
        String resolved = config.resolvedCacheDir().getPath();
        assertTrue(resolved.endsWith("my/cache") || resolved.endsWith("my\\cache"),
                "Expected path ending with my/cache but got: " + resolved);
    }
}
