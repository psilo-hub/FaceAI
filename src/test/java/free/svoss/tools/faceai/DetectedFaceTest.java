package free.svoss.tools.faceai;

import org.junit.jupiter.api.Test;

import java.awt.Rectangle;
import java.awt.image.BufferedImage;

import static org.junit.jupiter.api.Assertions.*;

class DetectedFaceTest {

    @Test
    void boundsReturnsCorrectRectangle() {
        DetectedFace face = new DetectedFace(10, 20, 100, 150, 0.95f);
        Rectangle bounds = face.bounds();
        assertEquals(new Rectangle(10, 20, 100, 150), bounds);
    }

    @Test
    void gettersReturnConstructorValues() {
        DetectedFace face = new DetectedFace(5, 10, 50, 60, 0.75f);
        assertEquals(5, face.x());
        assertEquals(10, face.y());
        assertEquals(50, face.width());
        assertEquals(60, face.height());
        assertEquals(0.75f, face.confidence());
    }

    @Test
    void cropReturnsCorrectSubimage() {
        BufferedImage img = new BufferedImage(200, 200, BufferedImage.TYPE_INT_RGB);
        DetectedFace face = new DetectedFace(10, 20, 50, 60, 0.9f);
        BufferedImage cropped = face.crop(img);
        assertEquals(50, cropped.getWidth());
        assertEquals(60, cropped.getHeight());
    }

    @Test
    void cropClampsToImageBounds() {
        BufferedImage img = new BufferedImage(100, 100, BufferedImage.TYPE_INT_RGB);
        // Face extends beyond image
        DetectedFace face = new DetectedFace(80, 80, 50, 50, 0.9f);
        BufferedImage cropped = face.crop(img);
        assertEquals(20, cropped.getWidth());  // clamped from 50 to 20
        assertEquals(20, cropped.getHeight());
    }

    @Test
    void cropThrowsOnNullSource() {
        DetectedFace face = new DetectedFace(0, 0, 10, 10, 0.5f);
        assertThrows(NullPointerException.class, () -> face.crop(null));
    }

    @Test
    void equalsAndHashCode() {
        DetectedFace a = new DetectedFace(1, 2, 3, 4, 0.5f);
        DetectedFace b = new DetectedFace(1, 2, 3, 4, 0.5f);
        DetectedFace c = new DetectedFace(1, 2, 3, 4, 0.6f);
        assertEquals(a, b);
        assertEquals(a.hashCode(), b.hashCode());
        assertNotEquals(a, c);
    }

    @Test
    void toStringContainsFields() {
        DetectedFace face = new DetectedFace(1, 2, 3, 4, 0.5f);
        String s = face.toString();
        assertTrue(s.contains("1"));
        assertTrue(s.contains("0.5"));
    }

    // ── Landmarks ───────────────────────────────────────────────────────

    /** Builds a 10-float landmark array with the two eyes at (x0,y0) and (x1,y1). */
    private static float[] eyes(double x0, double y0, double x1, double y1) {
        return new float[]{
                (float) x0, (float) y0,
                (float) x1, (float) y1,
                0f, 0f, 0f, 0f, 0f, 0f
        };
    }

    @Test
    void fiveArgConstructorHasNoLandmarks() {
        DetectedFace face = new DetectedFace(1, 2, 3, 4, 0.5f);
        assertFalse(face.hasLandmarks());
        assertNull(face.landmarks());
        assertTrue(Double.isNaN(face.rollAngleDegrees()));
    }

    @Test
    void sixArgConstructorExposesLandmarks() {
        float[] lm = eyes(0, 0, 10, 0);
        DetectedFace face = new DetectedFace(1, 2, 3, 4, 0.5f, lm);
        assertTrue(face.hasLandmarks());
        assertArrayEquals(lm, face.landmarks(), 1e-6f);
    }

    @Test
    void landmarksReturnsDefensiveCopy() {
        float[] source = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        DetectedFace face = new DetectedFace(0, 0, 1, 1, 0.5f, source);

        float[] first = face.landmarks();
        assertNotSame(source, first);
        first[0] = 99f;
        assertEquals(1f, face.landmarks()[0], 1e-6f, "mutating the returned array must not change state");

        source[0] = 42f;
        assertEquals(1f, face.landmarks()[0], 1e-6f, "mutating the source array must not change state");
    }

    @Test
    void rollAngleZeroForHorizontalEyes() {
        DetectedFace face = new DetectedFace(0, 0, 1, 1, 0.9f, eyes(0, 0, 10, 0));
        assertEquals(0.0, face.rollAngleDegrees(), 1e-6);
    }

    @Test
    void rollAnglePositiveNinetyForDownwardEyeLine() {
        DetectedFace face = new DetectedFace(0, 0, 1, 1, 0.9f, eyes(0, 0, 0, 10));
        assertEquals(90.0, face.rollAngleDegrees(), 1e-6);
    }

    @Test
    void rollAngleNegativeNinetyForUpwardEyeLine() {
        DetectedFace face = new DetectedFace(0, 0, 1, 1, 0.9f, eyes(0, 10, 0, 0));
        assertEquals(-90.0, face.rollAngleDegrees(), 1e-6);
    }

    @Test
    void rollAngleNormalisesReversedHorizontalToZero() {
        DetectedFace face = new DetectedFace(0, 0, 1, 1, 0.9f, eyes(10, 0, 0, 0));
        assertEquals(0.0, face.rollAngleDegrees(), 1e-6);
    }

    @Test
    void equalsAndHashCodeIncludeLandmarks() {
        DetectedFace a = new DetectedFace(1, 2, 3, 4, 0.5f, eyes(0, 0, 10, 0));
        DetectedFace b = new DetectedFace(1, 2, 3, 4, 0.5f, eyes(0, 0, 10, 0));
        DetectedFace c = new DetectedFace(1, 2, 3, 4, 0.5f, eyes(0, 0, 0, 10));
        DetectedFace noLandmarks = new DetectedFace(1, 2, 3, 4, 0.5f);

        assertEquals(a, b);
        assertEquals(a.hashCode(), b.hashCode());
        assertNotEquals(a, c);
        assertNotEquals(a, noLandmarks);
    }

    @Test
    void constructorRejectsWrongLandmarkLength() {
        assertThrows(IllegalArgumentException.class,
                () -> new DetectedFace(0, 0, 1, 1, 0.5f, new float[9]));
        assertThrows(IllegalArgumentException.class,
                () -> new DetectedFace(0, 0, 1, 1, 0.5f, new float[11]));
    }
}
