package free.svoss.tools.faceai.internal;

import java.util.List;
import java.util.Objects;

/**
 * Utility methods for embedding operations: cosine similarity and arithmetic mean.
 * <p>
 * All methods are pure Java — no DJL or model dependencies.
 */
public final class EmbeddingUtils {

    private EmbeddingUtils() {
        // utility class — no instantiation
    }

    /**
     * Computes cosine-based similarity between two embeddings, mapped to [0, 1].
     * <p>
     * Internally normalises both vectors, computes cosine similarity, then maps
     * from [-1, 1] to [0, 1] using {@code (cos + 1.0) / 2.0}.  The result is
     * clamped to [0.0, 1.0] to protect against floating-point drift.
     *
     * @param embeddingA first embedding (non-null, non-zero norm, same length as {@code embeddingB})
     * @param embeddingB second embedding (non-null, non-zero norm, same length as {@code embeddingA})
     * @return similarity in [0.0, 1.0] — 1.0 for identical direction, 0.5 for orthogonal, 0.0 for opposite
     * @throws NullPointerException     if either input is null
     * @throws IllegalArgumentException if the embeddings have different lengths or either is a zero vector
     */
    public static double calcSimilarity(float[] embeddingA, float[] embeddingB) {
        Objects.requireNonNull(embeddingA, "embeddingA must not be null");
        Objects.requireNonNull(embeddingB, "embeddingB must not be null");
        if (embeddingA.length != embeddingB.length) {
            throw new IllegalArgumentException(
                    "Embedding dimensions must match: " + embeddingA.length + " != " + embeddingB.length);
        }
        if (embeddingA.length == 0) {
            throw new IllegalArgumentException("Embeddings must not be empty");
        }

        double normA = 0.0;
        double normB = 0.0;
        double dot = 0.0;
        for (int i = 0; i < embeddingA.length; i++) {
            double a = embeddingA[i];
            double b = embeddingB[i];
            normA += a * a;
            normB += b * b;
            dot += a * b;
        }

        if (normA == 0.0) {
            throw new IllegalArgumentException("embeddingA is a zero vector");
        }
        if (normB == 0.0) {
            throw new IllegalArgumentException("embeddingB is a zero vector");
        }

        double cosine = dot / (Math.sqrt(normA) * Math.sqrt(normB));
        double similarity = (cosine + 1.0) / 2.0;
        return Math.max(0.0, Math.min(1.0, similarity));
    }

    /**
     * Computes the element-wise arithmetic mean of a list of embeddings.
     * <p>
     * The result is <strong>not</strong> L2-normalized.
     *
     * @param embeddings non-null, non-empty list of equal-length embeddings
     * @return a new float array containing the element-wise mean
     * @throws NullPointerException     if the list is null or contains a null element
     * @throws IllegalArgumentException if the list is empty or embeddings have different lengths
     */
    public static float[] calcAverage(List<float[]> embeddings) {
        Objects.requireNonNull(embeddings, "embeddings must not be null");
        if (embeddings.isEmpty()) {
            throw new IllegalArgumentException("embeddings must not be empty");
        }

        int dim = -1;
        for (float[] emb : embeddings) {
            Objects.requireNonNull(emb, "embeddings must not contain null elements");
            if (dim == -1) {
                dim = emb.length;
            } else if (emb.length != dim) {
                throw new IllegalArgumentException(
                        "All embeddings must have the same length: expected " + dim + " but got " + emb.length);
            }
        }

        double[] sum = new double[dim];
        for (float[] emb : embeddings) {
            for (int i = 0; i < dim; i++) {
                sum[i] += emb[i];
            }
        }

        float[] result = new float[dim];
        int count = embeddings.size();
        for (int i = 0; i < dim; i++) {
            result[i] = (float) (sum[i] / count);
        }
        return result;
    }
}
