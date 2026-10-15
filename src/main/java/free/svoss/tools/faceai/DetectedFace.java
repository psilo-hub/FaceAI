package free.svoss.tools.faceai;

import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.util.Objects;

/**
 * Value object representing a single detected face in an image.
 * <p>
 * A detection may optionally carry five facial landmarks (RetinaFace 5-point
 * convention: left eye, right eye, nose tip, left mouth corner, right mouth
 * corner) as pixel coordinates in the source image. When landmarks are present,
 * {@link #rollAngleDegrees()} exposes the in-plane rotation (roll) of the face.
 */
public final class DetectedFace {

    private final int x;
    private final int y;
    private final int width;
    private final int height;
    private final float confidence;

    /** 10 floats {@code (x0,y0,...,x4,y4)} in source pixels, or {@code null}. */
    private final float[] landmarks;

    /**
     * Creates a detection without landmarks.
     *
     * @param x          top-left x coordinate in pixels
     * @param y          top-left y coordinate in pixels
     * @param width      width in pixels
     * @param height     height in pixels
     * @param confidence confidence score between 0.0 and 1.0
     */
    public DetectedFace(int x, int y, int width, int height, float confidence) {
        this(x, y, width, height, confidence, null);
    }

    /**
     * Creates a detection, optionally carrying five facial landmarks.
     *
     * @param x          top-left x coordinate in pixels
     * @param y          top-left y coordinate in pixels
     * @param width      width in pixels
     * @param height     height in pixels
     * @param confidence confidence score between 0.0 and 1.0
     * @param landmarks  10 floats {@code (x0,y0,...,x4,y4)} in source-image pixel
     *                   coordinates, or {@code null} when unavailable. Points
     *                   {@code 0} and {@code 1} are the two eyes.
     * @throws IllegalArgumentException if {@code landmarks} is non-null and its
     *         length is not 10
     */
    public DetectedFace(int x, int y, int width, int height, float confidence, float[] landmarks) {
        if (landmarks != null && landmarks.length != 10) {
            throw new IllegalArgumentException("landmarks must have length 10");
        }
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.confidence = confidence;
        this.landmarks = landmarks == null ? null : landmarks.clone();
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

    /**
     * Returns {@code true} if this detection carries five-point landmarks.
     *
     * @return whether landmarks are available
     */
    public boolean hasLandmarks() {
        return landmarks != null;
    }

    /**
     * Returns a defensive copy of the five landmarks as 10 floats
     * {@code (x0,y0,...,x4,y4)} in source-image pixel coordinates.
     * Points {@code 0} and {@code 1} are the two eyes (the eye line).
     *
     * @return a copy of the landmark coordinates, or {@code null} if unavailable
     */
    public float[] landmarks() {
        return landmarks == null ? null : landmarks.clone();
    }

    /**
     * In-plane rotation (roll) of the face derived from the eye line
     * (landmark points {@code 0} and {@code 1}).
     * <p>
     * {@code 0.0} corresponds to an upright face (eye line horizontal);
     * positive values are clockwise. The returned angle is normalised to the
     * range {@code [-90, 90]} degrees.
     *
     * @return the roll angle in degrees, or {@link Double#NaN} when landmarks
     *         are unavailable
     */
    public double rollAngleDegrees() {
        if (landmarks == null) {
            return Double.NaN;
        }
        double x0 = landmarks[0];
        double y0 = landmarks[1];
        double x1 = landmarks[2];
        double y1 = landmarks[3];
        double theta = Math.toDegrees(Math.atan2(y1 - y0, x1 - x0)); // eyes 0 -> 1
        // Normalise so an upright face is ~0 and the value stays in [-90, 90].
        // The comparison is strict so that an exactly vertical eye line
        // (theta == -90) is reported as -90 rather than folded to +90.
        if (theta < -90.0) theta += 180.0;
        if (theta > 90.0) theta -= 180.0;
        return theta;
    }

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
                && height == that.height && Float.compare(that.confidence, confidence) == 0
                && Arrays.equals(landmarks, that.landmarks);
    }

    @Override
    public int hashCode() {
        return Objects.hash(x, y, width, height, confidence, Arrays.hashCode(landmarks));
    }

    @Override
    public String toString() {
        return "DetectedFace{x=" + x + ", y=" + y + ", width=" + width
                + ", height=" + height + ", confidence=" + confidence
                + ", landmarks=" + Arrays.toString(landmarks) + "}";
    }
}
