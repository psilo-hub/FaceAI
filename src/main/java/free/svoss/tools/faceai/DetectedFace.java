package free.svoss.tools.faceai;

import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.util.Objects;

/**
 * Value object representing a single detected face in an image.
 */
public final class DetectedFace {

    private final int x;
    private final int y;
    private final int width;
    private final int height;
    private final float confidence;

    public DetectedFace(int x, int y, int width, int height, float confidence) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.confidence = confidence;
    }

    /** Top-left x coordinate in pixels. */
    public int x() { return x; }
    /** Top-left y coordinate in pixels. */
    public int y() { return y; }
    /** Width in pixels. */
    public int width() { return width; }
    /** Height in pixels. */
    public int height() { return height; }
    /** Confidence score between 0.0 and 1.0. */
    public float confidence() { return confidence; }

    /** Returns a Rectangle representing the bounding box. */
    public Rectangle bounds() {
        return new Rectangle(x, y, width, height);
    }

    /**
     * Crops this face region from the source image.
     * Coordinates are clamped to the source image bounds.
     *
     * @param source the original image
     * @return the cropped sub-image containing the face
     * @throws NullPointerException if source is null
     */
    public BufferedImage crop(BufferedImage source) {
        Objects.requireNonNull(source, "source must not be null");
        int imgW = source.getWidth();
        int imgH = source.getHeight();
        int cx = Math.max(0, Math.min(x, imgW));
        int cy = Math.max(0, Math.min(y, imgH));
        int cw = Math.max(0, Math.min(width, imgW - cx));
        int ch = Math.max(0, Math.min(height, imgH - cy));
        if (cw == 0 || ch == 0) {
            throw new IllegalArgumentException("Face bounds are outside image dimensions");
        }
        return source.getSubimage(cx, cy, cw, ch);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof DetectedFace)) return false;
        DetectedFace that = (DetectedFace) o;
        return x == that.x && y == that.y && width == that.width
                && height == that.height && Float.compare(that.confidence, confidence) == 0;
    }

    @Override
    public int hashCode() {
        return Objects.hash(x, y, width, height, confidence);
    }

    @Override
    public String toString() {
        return "DetectedFace{x=" + x + ", y=" + y + ", width=" + width
                + ", height=" + height + ", confidence=" + confidence + "}";
    }
}
