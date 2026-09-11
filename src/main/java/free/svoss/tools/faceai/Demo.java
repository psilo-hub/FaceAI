package free.svoss.tools.faceai;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Command-line demo that showcases the FaceAI public API.
 * <p>
 * Loads the sample images from the {@code demo} folder (or the directory
 * given as the first argument), detects faces, extracts embeddings, and
 * reports pairwise similarities plus the average embedding. Runs the full
 * pipeline: <code>detect &#8594; crop &#8594; embed &#8594; compare</code>.
 *
 * <p>Run from the project root:
 * <pre>{@code
 * mvn -q compile exec:java -Dexec.mainClass=free.svoss.tools.faceai.Demo
 * // or, after mvn compile:
 * java -cp "target/classes;$(get-deps)" free.svoss.tools.faceai.Demo
 * }</pre>
 *
 * <p>The bundled sample images are:
 * <ul>
 *   <li>{@code demo/obama.jpg} and {@code demo/obama2.jpg} — the same person</li>
 *   <li>{@code demo/biden.jpg} — a different person</li>
 * </ul>
 */
public final class Demo {

    private Demo() {
        // utility class — not instantiated
    }

    public static void main(String[] args) throws IOException {
        File imageDir = new File(args.length > 0 ? args[0] : "demo");
        String[] imageNames = {"obama.jpg", "obama2.jpg", "biden.jpg"};

        try (FaceAI faceai = FaceAI.create()) {
            Map<String, float[]> embeddings = new LinkedHashMap<>();

            for (String name : imageNames) {
                File file = new File(imageDir, name);
                BufferedImage image = ImageIO.read(file);
                if (image == null) {
                    System.out.println("! could not read image: " + file.getPath());
                    continue;
                }
                System.out.println("== " + name + " (" + image.getWidth() + "x" + image.getHeight() + ") ==");

                // 1. Detect faces (sorted by descending confidence)
                DetectedFace[] faces = faceai.detectFaces(image);
                System.out.println("   detected " + faces.length + " face(s)");
                for (DetectedFace f : faces) {
                    System.out.printf("      x=%d y=%d w=%d h=%d confidence=%.4f%n",
                            f.x(), f.y(), f.width(), f.height(), f.confidence());
                }
                if (faces.length == 0) {
                    continue;
                }

                // 2. Crop the top-confidence face and compute its embedding
                DetectedFace top = faces[0];
                BufferedImage crop = top.crop(image);
                float[] embedding = faceai.getEmbedding(crop);
                embeddings.put(name, embedding);
                System.out.printf("   top face crop %dx%d %n", crop.getWidth(), crop.getHeight());
                System.out.printf("   embedding length=%d norm=%.4f%n", embedding.length, norm(embedding));
            }

            if (embeddings.size() < 2) {
                System.out.println("\nNot enough faces detected to compare embeddings.");
                return;
            }

            // 3. Compare every pair of embeddings
            System.out.println("\n-- Pairwise similarity (1.0 = identical, 0.5 = orthogonal, 0.0 = opposite) --");
            String[] names = embeddings.keySet().toArray(new String[0]);
            for (int i = 0; i < names.length; i++) {
                for (int j = i + 1; j < names.length; j++) {
                    double similarity = faceai.calcSimilarity(embeddings.get(names[i]), embeddings.get(names[j]));
                    System.out.printf("   %-12s <-> %-12s : %.4f%n", names[i], names[j], similarity);
                }
            }

            // 4. Average embedding (e.g. enrollment) and compare it back
            System.out.println("\n-- Average embedding --");
            float[] average = faceai.calcAverage(new ArrayList<>(embeddings.values()));
            System.out.printf("   average length=%d%n", average.length);
            for (Map.Entry<String, float[]> entry : embeddings.entrySet()) {
                System.out.printf("   similarity(average, %s) = %.4f%n",
                        entry.getKey(), faceai.calcSimilarity(average, entry.getValue()));
            }
            System.out.println("\nDone.");
        }
    }

    private static double norm(float[] vector) {
        double sum = 0.0;
        for (float v : vector) {
            sum += v * v;
        }
        return Math.sqrt(sum);
    }
}