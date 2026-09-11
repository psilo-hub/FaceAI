package free.svoss.tools.faceai.internal;

import ai.djl.modality.cv.Image;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import free.svoss.tools.faceai.DetectedFace;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * DJL Translator for RetinaFace face detection.
 * <p>
 * Converts a DJL {@link Image} into an array of {@link DetectedFace} objects
 * by running RetinaFace preprocessing, decoding bounding boxes from the raw
 * model output, applying confidence thresholding, and performing non-maximum
 * suppression (NMS).
 *
 * <p><strong>This class is internal — not part of the public API.</strong>
 */
public class FaceDetectionTranslator implements Translator<Image, DetectedFace[]> {

    private final double confThresh;
    private final double nmsThresh;
    private final int topK;
    private final double[] variance;
    private final int[][] scales;
    private final int[] steps;

    /**
     * @param confThresh confidence threshold (detections below this are discarded)
     * @param nmsThresh  IoU threshold for non-maximum suppression
     * @param variance   prior-box variance values [scaleXY, scaleWH]
     * @param topK       maximum number of detections to consider after thresholding
     * @param scales     anchor-box scales per feature level
     * @param steps      stride per feature level
     */
    public FaceDetectionTranslator(
            double confThresh,
            double nmsThresh,
            double[] variance,
            int topK,
            int[][] scales,
            int[] steps) {
        this.confThresh = confThresh;
        this.nmsThresh = nmsThresh;
        this.variance = variance;
        this.topK = topK;
        this.scales = scales;
        this.steps = steps;
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, Image input) {
        // Store original dimensions for post-processing
        ctx.setAttachment("width", input.getWidth());
        ctx.setAttachment("height", input.getHeight());

        NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.COLOR);
        // HWC → CHW, RGB → BGR
        array = array.transpose(2, 0, 1).flip(0);

        if (!array.getDataType().equals(DataType.FLOAT32)) {
            array = array.toType(DataType.FLOAT32, false);
        }

        // Subtract BGR mean (RetinaFace training used BGR with mean [104,117,123])
        NDArray mean = ctx.getNDManager()
                .create(new float[]{104f, 117f, 123f}, new Shape(3, 1, 1));
        array = array.sub(mean);

        return new NDList(array);
    }

    /** {@inheritDoc} */
    @Override
    public DetectedFace[] processOutput(TranslatorContext ctx, NDList list) {
        int width = (int) ctx.getAttachment("width");
        int height = (int) ctx.getAttachment("height");
        NDManager manager = ctx.getNDManager();

        double scaleXY = variance[0];
        double scaleWH = variance[1];

        // Extract face scores: list.get(1) has shape (N, 2) — [background, face]
        NDArray prob = list.get(1).get(":, 1:");
        prob = NDArrays.stack(
                new NDList(
                        prob.argMax(1).toType(DataType.FLOAT32, false),
                        prob.max(new int[]{1})));

        // Decode bounding boxes
        NDArray boxRecover = boxRecover(manager, width, height, scales, steps);
        NDArray boundingBoxes = list.get(0);
        NDArray bbWH = boundingBoxes.get(":, 2:")
                .mul(scaleWH).exp()
                .mul(boxRecover.get(":, 2:"));
        NDArray bbXY = boundingBoxes.get(":, :2")
                .mul(scaleXY)
                .mul(boxRecover.get(":, 2:"))
                .add(boxRecover.get(":, :2"))
                .sub(bbWH.mul(0.5f));
        boundingBoxes = NDArrays.concat(new NDList(bbXY, bbWH), 1);

        // Decode landmarks (5 points × 2 coords = 10 values per face)
        NDArray landms = list.get(2);
        landms = decodeLandm(landms, boxRecover, scaleXY);

        // Filter by confidence threshold
        NDArray cutOff = prob.get(1).gt(confThresh);
        boundingBoxes = boundingBoxes.transpose()
                .booleanMask(cutOff, 1).transpose();
        landms = landms.transpose().booleanMask(cutOff, 1).transpose();
        prob = prob.booleanMask(cutOff, 1);

        // Sort by probability, take topK
        long[] order = prob.get(1).argSort().get(":" + topK).toLongArray();
        prob = prob.transpose();

        List<DetectedFace> results = new ArrayList<>();
        Map<Integer, List<double[]>> recorder = new ConcurrentHashMap<>();

        for (int i = order.length - 1; i >= 0; i--) {
            long idx = order[i];
            float[] classProb = prob.get(idx).toFloatArray();
            int classId = (int) classProb[0];
            double probability = classProb[1];

            double[] boxArr = boundingBoxes.get(idx).toDoubleArray();

            // NMS: check IoU against already-accepted boxes for this class
            List<double[]> accepted = recorder.getOrDefault(classId, new ArrayList<>());
            boolean belowIoU = true;
            for (double[] acceptedBox : accepted) {
                if (computeIoU(boxArr, acceptedBox) > nmsThresh) {
                    belowIoU = false;
                    break;
                }
            }

            if (belowIoU) {
                // Convert normalized [x,y,w,h] to pixel coordinates
                double bx = boxArr[0] * width;
                double by = boxArr[1] * height;
                double bw = boxArr[2] * width;
                double bh = boxArr[3] * height;

                // Clamp to image bounds
                int ix = Math.max(0, (int) Math.round(bx));
                int iy = Math.max(0, (int) Math.round(by));
                int iw = Math.max(1, (int) Math.round(bw));
                int ih = Math.max(1, (int) Math.round(bh));
                if (ix + iw > width) {
                    iw = width - ix;
                }
                if (iy + ih > height) {
                    ih = height - iy;
                }

                if (iw > 0 && ih > 0) {
                    results.add(new DetectedFace(ix, iy, iw, ih, (float) probability));
                    accepted.add(boxArr);
                    recorder.put(classId, accepted);
                }
            }
        }

        // Sort by confidence descending
        results.sort((a, b) -> Float.compare(b.confidence(), a.confidence()));
        return results.toArray(new DetectedFace[0]);
    }

    // ── Helper methods ──────────────────────────────────────────────────

    /**
     * Generates default anchor boxes for the given image dimensions.
     */
    private NDArray boxRecover(
            NDManager manager, int width, int height, int[][] scales, int[] steps) {
        int[][] aspectRatio = new int[steps.length][2];
        for (int i = 0; i < steps.length; i++) {
            int wRatio = (int) Math.ceil((float) width / steps[i]);
            int hRatio = (int) Math.ceil((float) height / steps[i]);
            aspectRatio[i] = new int[]{hRatio, wRatio};
        }

        List<double[]> defaultBoxes = new ArrayList<>();
        for (int idx = 0; idx < steps.length; idx++) {
            int[] scale = scales[idx];
            for (int h = 0; h < aspectRatio[idx][0]; h++) {
                for (int w = 0; w < aspectRatio[idx][1]; w++) {
                    for (int s : scale) {
                        double skx = s * 1.0 / width;
                        double sky = s * 1.0 / height;
                        double cx = (w + 0.5) * steps[idx] / width;
                        double cy = (h + 0.5) * steps[idx] / height;
                        defaultBoxes.add(new double[]{cx, cy, skx, sky});
                    }
                }
            }
        }

        double[][] boxes = new double[defaultBoxes.size()][4];
        for (int i = 0; i < defaultBoxes.size(); i++) {
            boxes[i] = defaultBoxes.get(i);
        }
        return manager.create(boxes).clip(0.0, 1.0);
    }

    /**
     * Decodes 5-point face landmarks from model output.
     */
    private NDArray decodeLandm(NDArray pre, NDArray priors, double scaleXY) {
        NDArray point1 = pre.get(":, :2").mul(scaleXY)
                .mul(priors.get(":, 2:")).add(priors.get(":, :2"));
        NDArray point2 = pre.get(":, 2:4").mul(scaleXY)
                .mul(priors.get(":, 2:")).add(priors.get(":, :2"));
        NDArray point3 = pre.get(":, 4:6").mul(scaleXY)
                .mul(priors.get(":, 2:")).add(priors.get(":, :2"));
        NDArray point4 = pre.get(":, 6:8").mul(scaleXY)
                .mul(priors.get(":, 2:")).add(priors.get(":, :2"));
        NDArray point5 = pre.get(":, 8:10").mul(scaleXY)
                .mul(priors.get(":, 2:")).add(priors.get(":, :2"));
        return NDArrays.concat(new NDList(point1, point2, point3, point4, point5), 1);
    }

    /**
     * Computes IoU (Intersection over Union) between two boxes.
     * Each box is [x, y, w, h] in normalized coordinates.
     */
    static double computeIoU(double[] boxA, double[] boxB) {
        double aX1 = boxA[0];
        double aY1 = boxA[1];
        double aX2 = boxA[0] + boxA[2];
        double aY2 = boxA[1] + boxA[3];

        double bX1 = boxB[0];
        double bY1 = boxB[1];
        double bX2 = boxB[0] + boxB[2];
        double bY2 = boxB[1] + boxB[3];

        double interX1 = Math.max(aX1, bX1);
        double interY1 = Math.max(aY1, bY1);
        double interX2 = Math.min(aX2, bX2);
        double interY2 = Math.min(aY2, bY2);

        double interArea = Math.max(0, interX2 - interX1) * Math.max(0, interY2 - interY1);
        double areaA = boxA[2] * boxA[3];
        double areaB = boxB[2] * boxB[3];
        double unionArea = areaA + areaB - interArea;

        return unionArea > 0 ? interArea / unionArea : 0;
    }
}
