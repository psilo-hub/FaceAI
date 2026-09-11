package free.svoss.tools.faceai.internal;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import free.svoss.tools.faceai.DetectedFace;
import free.svoss.tools.faceai.FaceAIConfig;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * DJL Translator for RetinaFace face detection.
 * <p>
 * Preprocessing matches {@code Pytorch_Retinaface} exactly:
 * <ol>
 *   <li>Read pixels as BGR, values in {@code [0, 255]}</li>
 *   <li>Subtract the BGR channel-wise mean {@code [104, 117, 123]}</li>
 *   <li>Arrange as CHW tensor {@code 3 x H x W} (the leading batch dimension
 *       is added by DJL's batchifier)</li>
 * </ol>
 * <p>
 * Post-processing decodes anchor offsets from the raw model output
 * ({@code loc}, {@code conf}, {@code landms} tensors), applies the confidence
 * threshold, performs non-maximum suppression, clamps boxes and sorts by
 * descending confidence.
 *
 * <p><strong>This class is internal — not part of the public API.</strong>
 */
public final class RetinaFaceTranslator implements Translator<BufferedImage, DetectedFace[]> {

    // RetinaFace (ResNet-50 backbone) prior-box configuration.
    private static final double VARIANCE_IN = 0.1;
    private static final double VARIANCE_EN = 0.2;
    private static final int[][] SCALES = {{16, 32}, {64, 128}, {256, 512}};
    private static final int[] STEPS = {8, 16, 32};
    private static final int TOP_K = 5000;
    private static final float[] BGR_MEAN = {104f, 117f, 123f};

    private final FaceAIConfig config;

    /**
     * Creates a translator with the given configuration.
     *
     * @param config FaceAI config (provides thresholds)
     */
    public RetinaFaceTranslator(FaceAIConfig config) {
        if (config == null) {
            throw new NullPointerException("config must not be null");
        }
        this.config = config;
    }

    @Override
    public NDList processInput(TranslatorContext ctx, BufferedImage input) {
        // Store original dimensions for post-processing.
        ctx.setAttachment("width", (long) input.getWidth());
        ctx.setAttachment("height", (long) input.getHeight());

        int h = input.getHeight();
        int w = input.getWidth();
        float[] data = new float[3 * h * w];
        int idx = 0;
        for (int c = 0; c < 3; c++) {
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    int argb = input.getRGB(x, y);
                    int val;
                    if (c == 0) val = argb & 0xFF;               // B
                    else if (c == 1) val = (argb >> 8) & 0xFF;   // G
                    else val = (argb >> 16) & 0xFF;              // R
                    data[idx++] = val;
                }
            }
        }

        NDManager manager = ctx.getNDManager();
        NDArray array = manager.create(data, new Shape(3, h, w));
        // Subtract BGR mean (RetinaFace was trained on BGR input).
        NDArray mean = manager.create(BGR_MEAN, new Shape(3, 1, 1));
        return new NDList(array.sub(mean));
    }

    @Override
    public DetectedFace[] processOutput(TranslatorContext ctx, NDList list) {
        int width = (int) (long) ctx.getAttachment("width");
        int height = (int) (long) ctx.getAttachment("height");

        NDArray loc = list.get(0);   // [N, 4] anchor offsets
        NDArray conf = list.get(1);  // [N, 2] [background, face]
        long count = loc.getShape().get(0);

        float[] locData = loc.toFloatArray();
        float[] confData = conf.toFloatArray();

        double[] priorCx = new double[(int) count];
        double[] priorCy = new double[(int) count];
        double[] priorW = new double[(int) count];
        double[] priorH = new double[(int) count];
        generatePriors(width, height, priorCx, priorCy, priorW, priorH);

        float threshold = config.detectionThreshold();
        List<DetectedFace> candidates = new ArrayList<>();

        for (int i = 0; i < count; i++) {
            float score = confData[i * 2 + 1];
            if (score < threshold) {
                continue;
            }
            double cx = priorCx[i] + locData[i * 4] * VARIANCE_IN * priorW[i];
            double cy = priorCy[i] + locData[i * 4 + 1] * VARIANCE_IN * priorH[i];
            double bw = priorW[i] * Math.exp(locData[i * 4 + 2] * VARIANCE_EN);
            double bh = priorH[i] * Math.exp(locData[i * 4 + 3] * VARIANCE_EN);

            int x1 = Math.max(0, Math.min((int) Math.round((cx - bw / 2) * width), width));
            int y1 = Math.max(0, Math.min((int) Math.round((cy - bh / 2) * height), height));
            int x2 = Math.max(0, Math.min((int) Math.round((cx + bw / 2) * width), width));
            int y2 = Math.max(0, Math.min((int) Math.round((cy + bh / 2) * height), height));
            if (x2 - x1 > 0 && y2 - y1 > 0) {
                candidates.add(new DetectedFace(x1, y1, x2 - x1, y2 - y1, score));
            }
        }

        // Sort by confidence descending, keep top-K, then apply NMS.
        candidates.sort(Comparator.comparingDouble(DetectedFace::confidence).reversed());
        if (candidates.size() > TOP_K) {
            candidates = new ArrayList<>(candidates.subList(0, TOP_K));
        }
        return applyNms(candidates.toArray(new DetectedFace[0]), config.nmsThreshold());
    }

    /**
     * Generates RetinaFace prior boxes (normalized {@code [cx, cy, w, h]}) for
     * the given image dimensions, mirroring {@code Pytorch_Retinaface}.
     */
    static void generatePriors(int width, int height,
                               double[] outCx, double[] outCy,
                               double[] outW, double[] outH) {
        int idx = 0;
        for (int l = 0; l < STEPS.length; l++) {
            int step = STEPS[l];
            int gridY = (int) Math.ceil((float) height / step);
            int gridX = (int) Math.ceil((float) width / step);
            for (int yi = 0; yi < gridY; yi++) {
                for (int xi = 0; xi < gridX; xi++) {
                    for (int size : SCALES[l]) {
                        outCx[idx] = (xi + 0.5) * step / (double) width;
                        outCy[idx] = (yi + 0.5) * step / (double) height;
                        outW[idx] = size / (double) width;
                        outH[idx] = size / (double) height;
                        idx++;
                    }
                }
            }
        }
    }

    /**
     * Applies Non-Maximum Suppression to remove overlapping detections.
     *
     * @param faces        array of detected faces
     * @param iouThreshold IoU threshold for suppression
     * @return filtered array sorted by descending confidence
     */
    static DetectedFace[] applyNms(DetectedFace[] faces, float iouThreshold) {
        if (faces.length == 0) return faces;

        DetectedFace[] sorted = faces.clone();
        java.util.Arrays.sort(sorted,
                (a, b) -> Float.compare(b.confidence(), a.confidence()));

        boolean[] suppressed = new boolean[sorted.length];
        for (int i = 0; i < sorted.length; i++) {
            if (suppressed[i]) continue;
            for (int j = i + 1; j < sorted.length; j++) {
                if (suppressed[j]) continue;
                if (computeIoU(sorted[i], sorted[j]) > iouThreshold) {
                    suppressed[j] = true;
                }
            }
        }

        List<DetectedFace> result = new ArrayList<>();
        for (int i = 0; i < sorted.length; i++) {
            if (!suppressed[i]) {
                result.add(sorted[i]);
            }
        }
        return result.toArray(new DetectedFace[0]);
    }

    /**
     * Computes Intersection over Union (IoU) between two bounding boxes.
     */
    static float computeIoU(DetectedFace a, DetectedFace b) {
        int x1 = Math.max(a.x(), b.x());
        int y1 = Math.max(a.y(), b.y());
        int x2 = Math.min(a.x() + a.width(), b.x() + b.width());
        int y2 = Math.min(a.y() + a.height(), b.y() + b.height());

        int intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
        int union = a.width() * a.height() + b.width() * b.height() - intersection;

        return union > 0 ? (float) intersection / union : 0f;
    }
}