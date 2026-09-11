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
}
