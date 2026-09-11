package free.svoss.tools.faceai.internal;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;

/**
 * Image processing utilities for RGB conversion, alpha compositing, and resizing.
 * <p>
 * All methods are pure Java — no DJL or model dependencies.
 *
 * <p><strong>This class is internal — not part of the public API.</strong>
 */
public final class ImageUtils {

    private ImageUtils() {
        // utility class — no instantiation
    }

    /**
     * Converts the given image to {@link BufferedImage#TYPE_INT_RGB}.
     * <p>
     * If the image is already RGB it is returned as-is. If it has an
     * alpha channel the image is composited over a white background first.
     *
     * @param src source image (non-null)
     * @return an RGB image suitable for model input
     */
    public static BufferedImage toRgb(BufferedImage src) {
        if (src == null) {
            throw new NullPointerException("src must not be null");
        }
        if (src.getType() == BufferedImage.TYPE_INT_RGB) {
            return src;
        }
        // Images with alpha need compositing over white
        if (src.getColorModel().hasAlpha()) {
            src = alphaComposite(src);
        }
        // Convert to TYPE_INT_RGB
        BufferedImage rgb = new BufferedImage(
                src.getWidth(), src.getHeight(), BufferedImage.TYPE_INT_RGB);
        Graphics2D g = rgb.createGraphics();
        try {
            g.drawImage(src, 0, 0, null);
        } finally {
            g.dispose();
        }
        return rgb;
    }

    /**
     * Composites an image with alpha over a white background.
     *
     * @param src source image with alpha channel (non-null)
     * @return a new opaque RGB image
     */
    public static BufferedImage alphaComposite(BufferedImage src) {
        BufferedImage dest = new BufferedImage(
                src.getWidth(), src.getHeight(), BufferedImage.TYPE_INT_RGB);
        // Fill with white
        Graphics2D g = dest.createGraphics();
        try {
            g.setColor(java.awt.Color.WHITE);
            g.fillRect(0, 0, dest.getWidth(), dest.getHeight());
            g.drawImage(src, 0, 0, null);
        } finally {
            g.dispose();
        }
        return dest;
    }

    /**
     * Resizes the given image to the specified dimensions using bilinear interpolation.
     *
     * @param src         source image (non-null)
     * @param targetWidth  target width in pixels (must be positive)
     * @param targetHeight target height in pixels (must be positive)
     * @return a new resized image
     */
    public static BufferedImage resize(BufferedImage src, int targetWidth, int targetHeight) {
        if (src == null) {
            throw new NullPointerException("src must not be null");
        }
        if (targetWidth <= 0 || targetHeight <= 0) {
            throw new IllegalArgumentException(
                    "Target dimensions must be positive: " + targetWidth + "x" + targetHeight);
        }
        BufferedImage resized = new BufferedImage(
                targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = resized.createGraphics();
        try {
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                    RenderingHints.VALUE_INTERPOLATION_BILINEAR);
            g.setRenderingHint(RenderingHints.KEY_RENDERING,
                    RenderingHints.VALUE_RENDER_QUALITY);
            g.drawImage(src, 0, 0, targetWidth, targetHeight, null);
        } finally {
            g.dispose();
        }
        return resized;
    }

    /**
     * Letterbox resize: scales the image to fit within a square of the target
     * size preserving aspect ratio, then pads with gray (128,128,128).
     *
     * @param src        source image (non-null)
     * @param targetSize the square edge length (must be positive)
     * @return a square RGB image of {@code targetSize} x {@code targetSize}
     */
    public static BufferedImage letterbox(BufferedImage src, int targetSize) {
        if (src == null) {
            throw new NullPointerException("src must not be null");
        }
        if (targetSize <= 0) {
            throw new IllegalArgumentException("Target size must be positive: " + targetSize);
        }
        int w = src.getWidth();
        int h = src.getHeight();
        float scale = Math.min((float) targetSize / w, (float) targetSize / h);
        int newW = Math.max(1, Math.round(w * scale));
        int newH = Math.max(1, Math.round(h * scale));

        BufferedImage resized = resize(src, newW, newH);
        BufferedImage padded =
                new BufferedImage(targetSize, targetSize, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = padded.createGraphics();
        try {
            g.setColor(new java.awt.Color(128, 128, 128));
            g.fillRect(0, 0, targetSize, targetSize);
            int offsetX = (targetSize - newW) / 2;
            int offsetY = (targetSize - newH) / 2;
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                    RenderingHints.VALUE_INTERPOLATION_BILINEAR);
            g.drawImage(resized, offsetX, offsetY, null);
        } finally {
            g.dispose();
        }
        return padded;
    }

    /**
     * Extracts raw pixel values from an image in CHW order (channels first).
     * <p>
     * Values are in 0–255 range, in RGB channel order, row-major per channel:
     * {@code [channel][row][col]} flattened as {@code [c * H * W + y * W + x]}.
     * The image is converted to RGB first (alpha composited over white).
     *
     * @param img the source image (non-null)
     * @return float array of length 3 * H * W
     */
    public static float[] rgbPixels(BufferedImage img) {
        if (img == null) {
            throw new NullPointerException("img must not be null");
        }
        BufferedImage rgb = toRgb(img);
        int w = rgb.getWidth();
        int h = rgb.getHeight();
        float[] data = new float[3 * h * w];
        int idx = 0;
        for (int c = 0; c < 3; c++) {
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    int argb = rgb.getRGB(x, y);
                    int val;
                    if (c == 0) {
                        val = (argb >> 16) & 0xFF;
                    } else if (c == 1) {
                        val = (argb >> 8) & 0xFF;
                    } else {
                        val = argb & 0xFF;
                    }
                    data[idx++] = val;
                }
            }
        }
        return data;
    }

    /**
     * Converts a {@link BufferedImage} to an NCHW NDArray of shape
     * (1, 3, H, W) with pixel values normalized to [0, 1] (divide by 255).
     *
     * @param manager the NDManager used to create the array
     * @param img     the source image (non-null)
     * @return NDArray of shape (1, 3, H, W), float32, values in [0, 1]
     */
    public static NDArray toNDArray(NDManager manager, BufferedImage img) {
        if (manager == null) {
            throw new NullPointerException("manager must not be null");
        }
        BufferedImage rgb = toRgb(img);
        int h = rgb.getHeight();
        int w = rgb.getWidth();
        float[] pixels = rgbPixels(rgb);
        for (int i = 0; i < pixels.length; i++) {
            pixels[i] /= 255.0f;
        }
        return manager.create(pixels, new Shape(1, 3, h, w));
    }

    /**
     * Converts a {@link BufferedImage} to an NCHW NDArray with FaceNet
     * preprocessing applied: {@code (pixel / 255 - 0.5) / 0.5}.
     * <p>
     * This matches {@code facenet-pytorch} normalization and yields values in
     * [-1.0, 1.0]. The returned shape is (1, 3, H, W).
     *
     * @param manager the NDManager used to create the array
     * @param img     the source image (non-null)
     * @return NDArray of shape (1, 3, H, W), float32, FaceNet-normalized
     */
    public static NDArray toNDArrayNormalized(NDManager manager, BufferedImage img) {
        if (manager == null) {
            throw new NullPointerException("manager must not be null");
        }
        BufferedImage rgb = toRgb(img);
        int h = rgb.getHeight();
        int w = rgb.getWidth();
        float[] pixels = rgbPixels(rgb);
        for (int i = 0; i < pixels.length; i++) {
            pixels[i] = ((pixels[i] / 255.0f) - 0.5f) / 0.5f;
        }
        return manager.create(pixels, new Shape(1, 3, h, w));
    }
}
