package free.svoss.tools.faceai;

import free.svoss.tools.faceai.internal.EmbeddingUtils;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class EmbeddingUtilsTest {

    // ── calcSimilarity ────────────────────────────────────────────────────

    @Test
    void identicalVectorsReturnOne() {
        float[] a = {1f, 0f};
        float[] b = {1f, 0f};
        assertEquals(1.0, EmbeddingUtils.calcSimilarity(a, b), 1e-6);
    }

    @Test
    void orthogonalVectorsReturnHalf() {
        float[] a = {1f, 0f};
        float[] b = {0f, 1f};
        assertEquals(0.5, EmbeddingUtils.calcSimilarity(a, b), 1e-6);
    }

    @Test
    void oppositeVectorsReturnZero() {
        float[] a = {1f, 0f};
        float[] b = {-1f, 0f};
        assertEquals(0.0, EmbeddingUtils.calcSimilarity(a, b), 1e-6);
    }

    @Test
    void nonNormalizedSameDirectionReturnOne() {
        float[] a = {2f, 0f};
        float[] b = {3f, 0f};
        assertEquals(1.0, EmbeddingUtils.calcSimilarity(a, b), 1e-6);
    }

    @Test
    void nearIdenticalVectorsClampedToOne() {
        // Two vectors that are almost identical in direction.
        // cos(theta) can exceed 1.0 due to float rounding; clamping must apply.
        float eps = 1e-5f;
        float[] a = {1f, 0f, 0f};
        float[] b = {(float) Math.sqrt(1.0 - eps * eps), eps, 0f};
        // Manually check cosine to confirm it's within float rounding of 1.0
        double rawCos = a[0] * b[0] / (Math.sqrt(a[0] * a[0]) * Math.sqrt(b[0] * b[0] + b[1] * b[1]));
        double rawSimilarity = (rawCos + 1.0) / 2.0;
        // The mapped similarity should be clamped to exactly 1.0
        assertEquals(1.0, EmbeddingUtils.calcSimilarity(a, b), 1e-6,
                "Result should clamp to 1.0 for near-identical vectors (raw mapped: " + rawSimilarity + ")");
    }

    @Test
    void zeroVectorThrowsIllegalArgument() {
        float[] zero = {0f, 0f};
        float[] other = {1f, 0f};
        assertThrows(IllegalArgumentException.class, () -> EmbeddingUtils.calcSimilarity(zero, other));
        assertThrows(IllegalArgumentException.class, () -> EmbeddingUtils.calcSimilarity(other, zero));
    }

    @Test
    void nullInputsThrowNullPointer() {
        float[] a = {1f, 0f};
        assertThrows(NullPointerException.class, () -> EmbeddingUtils.calcSimilarity(null, a));
        assertThrows(NullPointerException.class, () -> EmbeddingUtils.calcSimilarity(a, null));
    }

    @Test
    void dimensionMismatchThrowsIllegalArgument() {
        float[] a = {1f, 0f};
        float[] b = {1f, 0f, 0f};
        assertThrows(IllegalArgumentException.class, () -> EmbeddingUtils.calcSimilarity(a, b));
    }

    // ── calcAverage ───────────────────────────────────────────────────────

    @Test
    void averageOfTwoVectors() {
        List<float[]> embeddings = Arrays.asList(
                new float[]{1f, 0f},
                new float[]{0f, 1f}
        );
        float[] avg = EmbeddingUtils.calcAverage(embeddings);
        assertArrayEquals(new float[]{0.5f, 0.5f}, avg, 1e-6f);
    }

    @Test
    void emptyListThrowsIllegalArgument() {
        List<float[]> empty = Collections.emptyList();
        assertThrows(IllegalArgumentException.class, () -> EmbeddingUtils.calcAverage(empty));
    }

    @Test
    void nullListThrowsNullPointer() {
        assertThrows(NullPointerException.class, () -> EmbeddingUtils.calcAverage(null));
    }

    @Test
    void listContainingNullThrowsNullPointer() {
        List<float[]> list = new ArrayList<>();
        list.add(new float[]{1f, 2f});
        list.add(null);
        assertThrows(NullPointerException.class, () -> EmbeddingUtils.calcAverage(list));
    }

    @Test
    void averageDimensionMismatchThrowsIllegalArgument() {
        List<float[]> list = Arrays.asList(
                new float[]{1f, 2f},
                new float[]{3f, 4f, 5f}
        );
        assertThrows(IllegalArgumentException.class, () -> EmbeddingUtils.calcAverage(list));
    }

    @Test
    void singleEmbeddingReturnsCopy() {
        float[] input = {1f, 2f, 3f};
        List<float[]> list = Collections.singletonList(input);
        float[] result = EmbeddingUtils.calcAverage(list);
        assertArrayEquals(input, result, 1e-6f);
        assertNotSame(input, result, "Must return a new array, not the input by reference");
    }

    @Test
    void averageOfThreeVectors() {
        List<float[]> embeddings = Arrays.asList(
                new float[]{3f, 6f},
                new float[]{0f, 0f},
                new float[]{3f, 6f}
        );
        float[] avg = EmbeddingUtils.calcAverage(embeddings);
        assertArrayEquals(new float[]{2f, 4f}, avg, 1e-6f);
    }

    @Test
    void averagePreservesDimensionality() {
        List<float[]> embeddings = Arrays.asList(
                new float[]{1f, 2f, 3f, 4f, 5f},
                new float[]{5f, 4f, 3f, 2f, 1f}
        );
        float[] avg = EmbeddingUtils.calcAverage(embeddings);
        assertEquals(5, avg.length);
        assertArrayEquals(new float[]{3f, 3f, 3f, 3f, 3f}, avg, 1e-6f);
    }
}
