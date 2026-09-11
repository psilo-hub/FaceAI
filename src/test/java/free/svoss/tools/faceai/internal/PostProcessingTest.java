package free.svoss.tools.faceai.internal;

import free.svoss.tools.faceai.DetectedFace;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class PostProcessingTest {

    @Test
    void computeIoUNoOverlap() {
        float[] a = {0, 0, 10, 10};
        float[] b = {20, 20, 30, 30};
        assertEquals(0.0, PostProcessing.computeIoU(a, b), 1e-6);
    }

    @Test
    void computeIoUCompleteOverlap() {
        float[] a = {0, 0, 10, 10};
        float[] b = {0, 0, 10, 10};
        assertEquals(1.0, PostProcessing.computeIoU(a, b), 1e-6);
    }

    @Test
    void computeIoUPartialOverlap() {
        float[] a = {0, 0, 10, 10};
        float[] b = {5, 5, 15, 15};
        // Intersection: 5x5=25, Union: 100+100-25=175, IoU=25/175=1/7
        assertEquals(1.0 / 7.0, PostProcessing.computeIoU(a, b), 1e-6);
    }

    @Test
    void nmsSuppressesOverlappingBoxes() {
        float[][] boxes = {
            {0, 0, 10, 10},
            {1, 1, 11, 11},  // high overlap with box 0
            {20, 20, 30, 30} // no overlap
        };
        List<Integer> indices = Arrays.asList(0, 1, 2);
        List<Integer> kept = PostProcessing.nms(boxes, indices, 0.5f);
        assertEquals(2, kept.size());
        assertTrue(kept.contains(0));
        assertTrue(kept.contains(2));
        assertFalse(kept.contains(1));
    }

    @Test
    void nmsKeepsNonOverlapping() {
        float[][] boxes = {
            {0, 0, 10, 10},
            {20, 20, 30, 30},
            {40, 40, 50, 50}
        };
        List<Integer> indices = Arrays.asList(0, 1, 2);
        List<Integer> kept = PostProcessing.nms(boxes, indices, 0.5f);
        assertEquals(3, kept.size());
    }

    @Test
    void processFiltersByConfidenceThreshold() {
        float[][] boxes = {
            {0, 0, 10, 10},
            {20, 20, 30, 30}
        };
        float[] scores = {0.9f, 0.3f};
        DetectedFace[] result = PostProcessing.process(boxes, scores, 0.5f, 0.4f, 100, 100);
        assertEquals(1, result.length);
        assertEquals(0.9f, result[0].confidence(), 0.001f);
    }

    @Test
    void processReturnsEmptyForNoFaces() {
        float[][] boxes = {{0, 0, 10, 10}};
        float[] scores = {0.1f};
        DetectedFace[] result = PostProcessing.process(boxes, scores, 0.5f, 0.4f, 100, 100);
        assertEquals(0, result.length);
    }

    @Test
    void processClampsToImageBounds() {
        float[][] boxes = {
            {-5, -5, 15, 15},  // extends outside
            {90, 90, 110, 110} // extends outside other side
        };
        float[] scores = {0.9f, 0.9f};
        DetectedFace[] result = PostProcessing.process(boxes, scores, 0.5f, 0.4f, 100, 100);
        assertEquals(2, result.length);
        for (DetectedFace face : result) {
            assertTrue(face.x() >= 0);
            assertTrue(face.y() >= 0);
            assertTrue(face.x() + face.width() <= 100);
            assertTrue(face.y() + face.height() <= 100);
        }
    }

    @Test
    void processSortsByDescendingConfidence() {
        float[][] boxes = {
            {0, 0, 10, 10},
            {20, 20, 30, 30},
            {40, 40, 50, 50}
        };
        float[] scores = {0.5f, 0.9f, 0.7f};
        DetectedFace[] result = PostProcessing.process(boxes, scores, 0.3f, 0.4f, 100, 100);
        assertEquals(3, result.length);
        assertTrue(result[0].confidence() >= result[1].confidence());
        assertTrue(result[1].confidence() >= result[2].confidence());
    }

    @Test
    void decodeBoxesProducesCorrectCoordinates() {
        float[][] offsets = {{0, 0, 0, 0}};
        float[][] anchors = {{0, 0, 10, 10}};
        float[][] decoded = PostProcessing.decodeBoxes(offsets, anchors);
        // With zero offsets, decoded should match anchor
        assertEquals(0f, decoded[0][0], 1e-6f);
        assertEquals(0f, decoded[0][1], 1e-6f);
        assertEquals(10f, decoded[0][2], 1e-6f);
        assertEquals(10f, decoded[0][3], 1e-6f);
    }
}
