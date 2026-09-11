package free.svoss.tools.faceai.internal;

import free.svoss.tools.faceai.DetectedFace;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * Post-processing utilities for face detection output.
 * <p>
 * Implements confidence thresholding, Non-Maximum Suppression (NMS),
 * box clamping, and sorting.
 *
 * <p><strong>This class is internal — not part of the public API.</strong>
 */
public final class PostProcessing {

    private PostProcessing() {
        // utility class
    }

    /**
     * Full post-processing pipeline for detection output.
     *
     * @param boxes     bounding boxes [N][4] as (x1, y1, x2, y2)
     * @param scores    confidence scores [N]
     * @param threshold confidence threshold
     * @param nmsThresh NMS IoU threshold
     * @param imgWidth  image width for clamping
     * @param imgHeight image height for clamping
     * @return DetectedFace[] sorted by descending confidence
     */
    public static DetectedFace[] process(float[][] boxes, float[] scores,
                                          float threshold, float nmsThresh,
                                          int imgWidth, int imgHeight) {
        // 1. Filter by confidence threshold
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < scores.length; i++) {
            if (scores[i] >= threshold) {
                indices.add(i);
            }
        }

        if (indices.isEmpty()) {
            return new DetectedFace[0];
        }

        // 2. Apply NMS
        List<Integer> kept = nms(boxes, indices, nmsThresh);

        // 3. Convert to DetectedFace with clamping
        List<DetectedFace> faces = new ArrayList<>();
        for (int idx : kept) {
            int x1 = Math.max(0, Math.min(Math.round(boxes[idx][0]), imgWidth));
            int y1 = Math.max(0, Math.min(Math.round(boxes[idx][1]), imgHeight));
            int x2 = Math.max(0, Math.min(Math.round(boxes[idx][2]), imgWidth));
            int y2 = Math.max(0, Math.min(Math.round(boxes[idx][3]), imgHeight));

            int w = x2 - x1;
            int h = y2 - y1;
            if (w > 0 && h > 0) {
                faces.add(new DetectedFace(x1, y1, w, h, scores[idx]));
            }
        }

        // 4. Sort by confidence descending
        Collections.sort(faces, Comparator.comparingDouble(DetectedFace::confidence).reversed());
        return faces.toArray(new DetectedFace[0]);
    }

    /**
     * Non-Maximum Suppression.
     *
     * @param boxes   bounding boxes [N][4] as (x1, y1, x2, y2)
     * @param indices candidate indices
     * @param iouThresh IoU threshold
     * @return list of kept indices
     */
    public static List<Integer> nms(float[][] boxes, List<Integer> indices, float iouThresh) {
        List<Integer> kept = new ArrayList<>(indices);
        boolean[] suppressed = new boolean[indices.size()];

        for (int i = 0; i < kept.size(); i++) {
            if (suppressed[i]) continue;
            int a = kept.get(i);
            for (int j = i + 1; j < kept.size(); j++) {
                if (suppressed[j]) continue;
                int b = kept.get(j);
                double iou = computeIoU(boxes[a], boxes[b]);
                if (iou > iouThresh) {
                    suppressed[j] = true;
                }
            }
        }

        List<Integer> result = new ArrayList<>();
        for (int i = 0; i < kept.size(); i++) {
            if (!suppressed[i]) {
                result.add(kept.get(i));
            }
        }
        return result;
    }

    /**
     * Computes Intersection over Union (IoU) for two axis-aligned bounding boxes.
     *
     * @param boxA [x1, y1, x2, y2]
     * @param boxB [x1, y1, x2, y2]
     * @return IoU value in [0.0, 1.0]
     */
    public static double computeIoU(float[] boxA, float[] boxB) {
        float x1 = Math.max(boxA[0], boxB[0]);
        float y1 = Math.max(boxA[1], boxB[1]);
        float x2 = Math.min(boxA[2], boxB[2]);
        float y2 = Math.min(boxA[3], boxB[3]);

        float interArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
        float areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);
        float areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]);
        float unionArea = areaA + areaB - interArea;

        if (unionArea <= 0) return 0.0;
        return interArea / unionArea;
    }

    /**
     * Decodes anchor-based bounding box offsets to absolute coordinates.
     *
     * @param offsets raw model output offsets [N][4] (dx, dy, dw, dh)
     * @param anchors anchor boxes [N][4] (x1, y1, x2, y2)
     * @return decoded boxes [N][4] (x1, y1, x2, y2)
     */
    public static float[][] decodeBoxes(float[][] offsets, float[][] anchors) {
        int n = offsets.length;
        float[][] decoded = new float[n][4];
        for (int i = 0; i < n; i++) {
            float anchorW = anchors[i][2] - anchors[i][0];
            float anchorH = anchors[i][3] - anchors[i][1];
            float anchorCx = (anchors[i][0] + anchors[i][2]) / 2f;
            float anchorCy = (anchors[i][1] + anchors[i][3]) / 2f;

            float predCx = anchorCx + offsets[i][0] * anchorW;
            float predCy = anchorCy + offsets[i][1] * anchorH;
            float predW = anchorW * (float) Math.exp(offsets[i][2]);
            float predH = anchorH * (float) Math.exp(offsets[i][3]);

            decoded[i][0] = predCx - predW / 2f;
            decoded[i][1] = predCy - predH / 2f;
            decoded[i][2] = predCx + predW / 2f;
            decoded[i][3] = predCy + predH / 2f;
        }
        return decoded;
    }
}
