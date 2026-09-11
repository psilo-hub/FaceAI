package free.svoss.tools.faceai.internal;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import free.svoss.tools.faceai.DetectedFace;
import free.svoss.tools.faceai.FaceAIConfig;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

/**
 * DJL Translator for RetinaFace face detection.
 * <p>
 * Preprocessing: RGB conversion, letterbox resize to model input size
 * (default 640×640).  Postprocessing: parse bounding boxes and confidence
 * scores from model output, apply confidence threshold and NMS, clamp
 * boxes, sort by confidence descending.
 *
 * <p><strong>This class is internal — not part of the public API.</strong>
 */
public final class RetinaFaceTranslator implements Translator<BufferedImage, DetectedFace[]> {

    private final FaceAIConfig config;
    private final int inputSize;

    /**
     * Creates a translator with the given configuration.
     *
     * @param config FaceAI config (provides thresholds)
     */
    public RetinaFaceTranslator(FaceAIConfig config) {
        this.config = config;
        this.inputSize = ModelConstants.RETINAFACE_INPUT_SIZE;
    }

    @Override
    public NDList processInput(TranslatorContext ctx, BufferedImage input) {
        NDManager manager = ctx.getNDManager();
        BufferedImage rgb = ImageUtils.toRgb(input);
        BufferedImage resized = letterbox(rgb, inputSize);
        NDArray array = toNdArray(manager, resized);
        return new NDList(array);
    }

    @Override
    public DetectedFace[] processOutput(TranslatorContext ctx, NDList list) {
        NDArray output = list.singletonOrThrow();
        float threshold = config.detectionThreshold();

        List<DetectedFace> faces = new ArrayList<>();

        if (output.getShape().dimension() == 2) {
            long rows = output.getShape().get(0);
            long cols = output.getShape().get(1);
            float[] data = output.toType(DataType.FLOAT32, true).toFloatArray();

            for (int i = 0; i < rows; i++) {
                if (cols >= 5) {
                    float x1 = data[i * (int) cols];
                    float y1 = data[i * (int) cols + 1];
                    float x2 = data[i * (int) cols + 2];
                    float y2 = data[i * (int) cols + 3];
                    float score = data[i * (int) cols + 4];

                    if (score < threshold) continue;

                    int x = Math.max(0, Math.round(x1));
                    int y = Math.max(0, Math.round(y1));
                    int w = Math.max(1, Math.round(x2 - x1));
                    int h = Math.max(1, Math.round(y2 - y1));

                    // Clamp to model input bounds
                    if (x + w > inputSize) w = inputSize - x;
                    if (y + h > inputSize) h = inputSize - y;
                    if (w <= 0 || h <= 0) continue;

                    faces.add(new DetectedFace(x, y, w, h, score));
                }
            }
        }

        return applyNms(faces.toArray(new DetectedFace[0]), config.nmsThreshold());
    }

    /**
     * Letterboxes an image to exactly {@code targetSize} × {@code targetSize}.
     * <p>
     * The source is scaled (aspect ratio preserved) to fit inside the target
     * square, centered, and padded with WHITE. Returns a square RGB image.
     */
    private static BufferedImage letterbox(BufferedImage src, int targetSize) {
        int srcW = src.getWidth();
        int srcH = src.getHeight();
        float scale = Math.min((float) targetSize / srcW, (float) targetSize / srcH);
        int scaledW = Math.max(1, Math.round(srcW * scale));
        int scaledH = Math.max(1, Math.round(srcH * scale));
        int padX = (targetSize - scaledW) / 2;
        int padY = (targetSize - scaledH) / 2;

        BufferedImage out = new BufferedImage(targetSize, targetSize, BufferedImage.TYPE_INT_RGB);
        java.awt.Graphics2D g = out.createGraphics();
        try {
            g.setColor(java.awt.Color.WHITE);
            g.fillRect(0, 0, targetSize, targetSize);
            g.setRenderingHint(java.awt.RenderingHints.KEY_INTERPOLATION,
                    java.awt.RenderingHints.VALUE_INTERPOLATION_BILINEAR);
            g.drawImage(src, padX, padY, scaledW, scaledH, null);
        } finally {
            g.dispose();
        }
        return out;
    }

    /**
     * Converts a BufferedImage to an NDArray in NCHW format (1×3×H×W)
     * with pixel values scaled to [0, 1].
     */
    private static NDArray toNdArray(NDManager manager, BufferedImage img) {
        int w = img.getWidth();
        int h = img.getHeight();
        float[] data = new float[3 * h * w];
        int idx = 0;
        for (int c = 0; c < 3; c++) {
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    int rgb = img.getRGB(x, y);
                    int val;
                    if (c == 0) val = (rgb >> 16) & 0xFF;
                    else if (c == 1) val = (rgb >> 8) & 0xFF;
                    else val = rgb & 0xFF;
                    data[idx++] = val / 255.0f;
                }
            }
        }
        return manager.create(data, new ai.djl.ndarray.types.Shape(1, 3, h, w));
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