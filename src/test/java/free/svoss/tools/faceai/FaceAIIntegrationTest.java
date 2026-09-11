package free.svoss.tools.faceai;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * End-to-end integration test for the full FaceAI pipeline:
 * detection → crop → embedding → similarity.
 * <p>
 * Tagged {@code integration} — excluded from the default test run
 * ({@code mvn test}) and executed via {@code mvn verify -Pintegration}.
 * Downloads the RetinaFace and FaceNet models on first run
 * (cached in the DJL cache directory afterwards).
 */
@Tag("integration")
class FaceAIIntegrationTest {

    private BufferedImage loadImage(String name) throws IOException {
        try (InputStream in = getClass().getResourceAsStream("/faces/" + name)) {
            assertNotNull(in, "Missing test image: " + name);
            return ImageIO.read(in);
        }
    }

    @Test
    void fullPipelineDetectsEmbedsAndCompares() throws IOException {
        BufferedImage imgA = loadImage("person1_a.jpg");
        BufferedImage imgB = loadImage("person1_b.jpg");
        BufferedImage imgC = loadImage("person2.jpg");

        try (FaceAI faceai = FaceAI.create()) {
            // Detection
            DetectedFace[] facesA = faceai.detectFaces(imgA);
            DetectedFace[] facesB = faceai.detectFaces(imgB);
            DetectedFace[] facesC = faceai.detectFaces(imgC);

            assertNotNull(facesA, "detectFaces must never return null");
            assertNotNull(facesB, "detectFaces must never return null");
            assertNotNull(facesC, "detectFaces must never return null");

            // The sample images each contain one face; models may vary in
            // strictness, so we require at least one detection per image.
            assertTrue(facesA.length >= 1, "person1_a should yield at least one face");
            assertTrue(facesB.length >= 1, "person1_b should yield at least one face");
            assertTrue(facesC.length >= 1, "person2 should yield at least one face");

            // Embeddings from top-confidence detections
            float[] embA = faceai.getEmbedding(facesA[0].crop(imgA));
            float[] embB = faceai.getEmbedding(facesB[0].crop(imgB));
            float[] embC = faceai.getEmbedding(facesC[0].crop(imgC));

            assertEquals(512, embA.length, "FaceNet embedding should be 512-d");
            assertEquals(512, embB.length, "FaceNet embedding should be 512-d");
            assertEquals(512, embC.length, "FaceNet embedding should be 512-d");

            // Same person (person1_a vs person1_b) should be more similar
            // than different people (person1_a vs person2).
            double same = faceai.calcSimilarity(embA, embB);
            double diff = faceai.calcSimilarity(embA, embC);

            assertTrue(same >= 0.0 && same <= 1.0, "similarity must be in [0,1]");
            assertTrue(diff >= 0.0 && diff <= 1.0, "similarity must be in [0,1]");
            assertTrue(same > diff,
                    "same-person similarity (" + same + ") should exceed different-person (" + diff + ")");

            // Average embedding retains dimension and yields valid similarity
            float[] avg = faceai.calcAverage(java.util.List.of(embA, embB, embC));
            assertEquals(512, avg.length, "Average embedding should be 512-d");
            double avgSim = faceai.calcSimilarity(avg, embA);
            assertTrue(avgSim >= 0.0 && avgSim <= 1.0, "avg similarity must be in [0,1]");
        }
    }

    @Test
    void detectFacesValidatesInputs() throws IOException {
        try (FaceAI faceai = FaceAI.create()) {
            assertThrows(NullPointerException.class, () -> faceai.detectFaces(null));
            BufferedImage empty = new BufferedImage(0, 0, BufferedImage.TYPE_INT_RGB);
            assertThrows(IllegalArgumentException.class, () -> faceai.detectFaces(empty));
        }
    }
}