package free.svoss.tools.faceai.internal;

import free.svoss.tools.faceai.DetectedFace;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class RetinaFaceTranslatorTest {

    // ── IoU computation ─────────────────────────────────────────────────

    @Test
    void iouOfIdenticalBoxesIsOne() {
        DetectedFace a = new DetectedFace(10, 10, 50, 50, 0.9f);
        float iou = RetinaFaceTranslator.computeIoU(a, a);
        assertEquals(1.0f, iou, 1e-6f);
    }

    @Test
    void iouOfNonOverlappingBoxesIsZero() {
        DetectedFace a = new DetectedFace(0, 0, 10, 10, 0.9f);
        DetectedFace b = new DetectedFace(100, 100, 10, 10, 0.8f);
        float iou = RetinaFaceTranslator.computeIoU(a, b);
        assertEquals(0f, iou, 1e-6f);
    }

    @Test
    void iouOfPartiallyOverlappingBoxes() {
        // Two 10x10 boxes with 5x5 overlap
        DetectedFace a = new DetectedFace(0, 0, 10, 10, 0.9f);
        DetectedFace b = new DetectedFace(5, 5, 10, 10, 0.8f);
        float iou = RetinaFaceTranslator.computeIoU(a, b);
        // Intersection = 5*5 = 25, Union = 100+100-25 = 175
        assertEquals(25f / 175f, iou, 1e-5f);
    }

    // ── NMS ─────────────────────────────────────────────────────────────

    @Test
    void nmsRemovesOverlappingDuplicates() {
        DetectedFace[] faces = {
            new DetectedFace(10, 10, 50, 50, 0.9f),
            new DetectedFace(12, 12, 50, 50, 0.85f), // high overlap with first
            new DetectedFace(200, 200, 30, 30, 0.7f)  // no overlap
        };
        DetectedFace[] result = RetinaFaceTranslator.applyNms(faces, 0.5f);
        assertEquals(2, result.length, "Should keep 2 faces after NMS");
    }

    @Test
    void nmsKeepsAllNonOverlappingFaces() {
        DetectedFace[] faces = {
            new DetectedFace(0, 0, 10, 10, 0.9f),
            new DetectedFace(100, 100, 10, 10, 0.8f),
            new DetectedFace(200, 200, 10, 10, 0.7f)
        };
        DetectedFace[] result = RetinaFaceTranslator.applyNms(faces, 0.5f);
        assertEquals(3, result.length);
    }

    @Test
    void nmsReturnsEmptyForEmptyInput() {
        DetectedFace[] result = RetinaFaceTranslator.applyNms(new DetectedFace[0], 0.5f);
        assertEquals(0, result.length);
    }

    @Test
    void nmsKeepsHighestConfidenceWhenTwoOverlap() {
        DetectedFace[] faces = {
            new DetectedFace(0, 0, 50, 50, 0.7f),
            new DetectedFace(2, 2, 50, 50, 0.9f)  // almost identical, higher confidence
        };
        DetectedFace[] result = RetinaFaceTranslator.applyNms(faces, 0.3f);
        assertEquals(1, result.length);
        assertEquals(0.9f, result[0].confidence(), 1e-6f);
    }

    @Test
    void nmsResultSortedByConfidenceDescending() {
        DetectedFace[] faces = {
            new DetectedFace(0, 0, 10, 10, 0.5f),
            new DetectedFace(100, 100, 10, 10, 0.9f),
            new DetectedFace(200, 200, 10, 10, 0.7f)
        };
        DetectedFace[] result = RetinaFaceTranslator.applyNms(faces, 0.5f);
        assertTrue(result.length >= 2);
        for (int i = 0; i < result.length - 1; i++) {
            assertTrue(result[i].confidence() >= result[i + 1].confidence(),
                    "Results should be sorted by descending confidence");
        }
    }

    // ── Landmark decoding ───────────────────────────────────────────────

    private static final int W = 64;
    private static final int H = 64;
    private static final int COUNT = 168; // 8x8x2 + 4x4x2 + 2x2x2 for a 64x64 image

    private static double[][] priors() {
        double[] cx = new double[COUNT];
        double[] cy = new double[COUNT];
        double[] w = new double[COUNT];
        double[] h = new double[COUNT];
        RetinaFaceTranslator.generatePriors(W, H, cx, cy, w, h);
        return new double[][]{cx, cy, w, h};
    }

    @Test
    void decodeLandmarksUsesPriorCentreForZeroOffsets() {
        double[][] p = priors();
        float[] raw = new float[COUNT * 10];
        float[] lm = RetinaFaceTranslator.decodeLandmarks(raw, 0, p[0], p[1], p[2], p[3], W, H);

        assertEquals(10, lm.length);
        // First prior centre is (0.5*8)/64 = 0.0625 -> 4 px on both axes.
        for (float v : lm) {
            assertEquals(4f, v, 1e-4f);
        }
    }

    @Test
    void decodeLandmarksAppliesOffsetAndPriorScale() {
        double[][] p = priors();
        float[] raw = new float[COUNT * 10];
        raw[0] = 1.0f; // x offset of point 0
        float[] lm = RetinaFaceTranslator.decodeLandmarks(raw, 0, p[0], p[1], p[2], p[3], W, H);

        // lx = 0.0625 + 1.0 * 0.1 * 0.25 = 0.0875 -> 5.6 px -> 6
        assertEquals(6f, lm[0], 1e-4f);
        // y offset is zero -> prior centre -> 4 px
        assertEquals(4f, lm[1], 1e-4f);
    }

    @Test
    void decodeLandmarksClampsToImageBounds() {
        double[][] p = priors();
        float[] raw = new float[COUNT * 10];
        raw[0] = 10000f;   // far right
        raw[1] = -10000f;  // far above
        float[] lm = RetinaFaceTranslator.decodeLandmarks(raw, 0, p[0], p[1], p[2], p[3], W, H);

        assertEquals(W, lm[0], 1e-4f);
        assertEquals(0f, lm[1], 1e-4f);
    }

    @Test
    void decodeLandmarksMapsAllFivePoints() {
        double[][] p = priors();
        float[] raw = new float[COUNT * 10];
        // Give point index 2 (nose) a distinct x offset.
        raw[4] = 1.0f;
        float[] lm = RetinaFaceTranslator.decodeLandmarks(raw, 0, p[0], p[1], p[2], p[3], W, H);

        assertEquals(6f, lm[4], 1e-4f);
        assertEquals(4f, lm[5], 1e-4f);
    }
}
