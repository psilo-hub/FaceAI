package free.svoss.tools.faceai.internal;

import org.junit.jupiter.api.Test;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;

import static org.junit.jupiter.api.Assertions.*;

class ImageUtilsTest {

    // ── toRgb ───────────────────────────────────────────────────────────

    @Test
    void toRgbConvertsGrayscale() {
        BufferedImage gray = new BufferedImage(4, 4, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g = gray.createGraphics();
        g.setColor(Color.RED);
        g.fillRect(0, 0, 4, 4);
        g.dispose();

        BufferedImage rgb = ImageUtils.toRgb(gray);
        assertEquals(BufferedImage.TYPE_INT_RGB, rgb.getType());
        assertEquals(4, rgb.getWidth());
        assertEquals(4, rgb.getHeight());
    }

    @Test
    void toRgbCompositesAlphaOverWhite() {
        BufferedImage argb = new BufferedImage(2, 2, BufferedImage.TYPE_INT_ARGB);
        // Fully transparent red — should appear as white after compositing
        argb.setRGB(0, 0, new Color(255, 0, 0, 0).getRGB());

        BufferedImage rgb = ImageUtils.toRgb(argb);
        int pixel = rgb.getRGB(0, 0) & 0xFFFFFF;
        int white = Color.WHITE.getRGB() & 0xFFFFFF;
        assertEquals(white, pixel, "Transparent pixel should composite to white");
    }

    @Test
    void toRgbReturnsSameInstanceForTypeIntRgb() {
        BufferedImage rgb = new BufferedImage(4, 4, BufferedImage.TYPE_INT_RGB);
        BufferedImage result = ImageUtils.toRgb(rgb);
        assertSame(rgb, result);
    }

    @Test
    void toRgbRejectsNull() {
        assertThrows(NullPointerException.class, () -> ImageUtils.toRgb(null));
    }

    @Test
    void toRgbNeverModifiesInput() {
        BufferedImage src = new BufferedImage(4, 4, BufferedImage.TYPE_INT_ARGB);
        int before = src.getRGB(0, 0);
        ImageUtils.toRgb(src);
        assertEquals(before, src.getRGB(0, 0), "Input must not be modified");
    }

    // ── resize ──────────────────────────────────────────────────────────

    @Test
    void resizeProducesCorrectDimensions() {
        BufferedImage src = new BufferedImage(100, 50, BufferedImage.TYPE_INT_RGB);
        BufferedImage resized = ImageUtils.resize(src, 200, 100);
        assertEquals(200, resized.getWidth());
        assertEquals(100, resized.getHeight());
    }

    @Test
    void resizeRejectsNull() {
        assertThrows(NullPointerException.class, () -> ImageUtils.resize(null, 10, 10));
    }

    @Test
    void resizeRejectsNonPositiveTarget() {
        BufferedImage src = new BufferedImage(10, 10, BufferedImage.TYPE_INT_RGB);
        assertThrows(IllegalArgumentException.class,
                () -> ImageUtils.resize(src, 0, 10));
        assertThrows(IllegalArgumentException.class,
                () -> ImageUtils.resize(src, 10, -5));
    }

    // ── alphaComposite ──────────────────────────────────────────────────

    @Test
    void alphaCompositeFillsWithWhite() {
        BufferedImage argb = new BufferedImage(2, 2, BufferedImage.TYPE_INT_ARGB);
        argb.setRGB(0, 0, new Color(255, 0, 0, 0).getRGB());

        BufferedImage dest = ImageUtils.alphaComposite(argb);
        int pixel = dest.getRGB(0, 0) & 0xFFFFFF;
        assertEquals(Color.WHITE.getRGB() & 0xFFFFFF, pixel);
    }

    @Test
    void alphaCompositeKeepsOpaquePixels() {
        BufferedImage argb = new BufferedImage(2, 2, BufferedImage.TYPE_INT_ARGB);
        argb.setRGB(0, 0, new Color(10, 20, 30, 255).getRGB());

        BufferedImage dest = ImageUtils.alphaComposite(argb);
        int pixel = dest.getRGB(0, 0) & 0xFFFFFF;
        int expected = new Color(10, 20, 30).getRGB() & 0xFFFFFF;
        assertEquals(expected, pixel);
    }
}