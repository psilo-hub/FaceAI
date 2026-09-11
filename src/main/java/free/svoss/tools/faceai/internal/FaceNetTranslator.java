package free.svoss.tools.faceai.internal;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import free.svoss.tools.faceai.FaceAIConfig;

import java.awt.image.BufferedImage;

/**
 * DJL Translator for FaceNet face recognition (feature extraction).
 * <p>
 * Preprocessing matches {@code facenet-pytorch} normalization exactly:
 * <ol>
 *   <li>Convert to RGB</li>
 *   <li>Resize to 160×160</li>
 *   <li>Scale pixel values to [0, 1] (divide by 255)</li>
 *   <li>Normalize: {@code (pixel - 0.5) / 0.5} → result range [-1.0, 1.0]</li>
 *   <li>Convert to NCHW tensor: 1×3×160×160</li>
 * </ol>
 * Model output: 1×512 embedding vector.  If
 * {@link FaceAIConfig#l2NormalizeEmbeddings()} is {@code true}, the output
 * vector is L2-normalized before returning.
 *
 * <p><strong>This class is internal — not part of the public API.</strong>
 */
public final class FaceNetTranslator implements Translator<BufferedImage, float[]> {

    private final FaceAIConfig config;
    private final int inputSize;

    /**
     * Creates a translator with the given configuration.
     *
     * @param config FaceAI config (provides embeddingDimension, l2NormalizeEmbeddings)
     */
    public FaceNetTranslator(FaceAIConfig config) {
        this.config = config;
        this.inputSize = ModelConstants.FACENET_INPUT_SIZE;
    }

    @Override
    public NDList processInput(TranslatorContext ctx, BufferedImage input) {
        NDManager manager = ctx.getNDManager();
        BufferedImage rgb = ImageUtils.toRgb(input);
        BufferedImage resized = ImageUtils.resize(rgb, inputSize, inputSize);

        int w = resized.getWidth();
        int h = resized.getHeight();
        float[] data = new float[3 * h * w];
        int idx = 0;
        for (int c = 0; c < 3; c++) {
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    int argb = resized.getRGB(x, y);
                    float val;
                    if (c == 0) val = ((argb >> 16) & 0xFF) / 255.0f;
                    else if (c == 1) val = ((argb >> 8) & 0xFF) / 255.0f;
                    else val = (argb & 0xFF) / 255.0f;
                    // facenet-pytorch: (pixel - 0.5) / 0.5 → [-1, 1]
                    data[idx++] = (val - 0.5f) / 0.5f;
                }
            }
        }
        NDArray array = manager.create(data, new ai.djl.ndarray.types.Shape(1, 3, h, w));
        return new NDList(array);
    }

    @Override
    public float[] processOutput(TranslatorContext ctx, NDList list) {
        NDArray embedding = list.singletonOrThrow();
        if (embedding.getShape().dimension() > 1) {
            embedding = embedding.flatten();
        }
        float[] result = embedding.toFloatArray();

        if (config.l2NormalizeEmbeddings()) {
            double norm = 0.0;
            for (float v : result) {
                norm += v * v;
            }
            norm = Math.sqrt(norm);
            if (norm > 0) {
                for (int i = 0; i < result.length; i++) {
                    result[i] = (float) (result[i] / norm);
                }
            }
        }

        return result;
    }
}