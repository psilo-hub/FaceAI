package free.svoss.tools.faceai.internal;

import free.svoss.tools.faceai.RecordingModelDownloadListener;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DownloadProgressTest {

    private static final String MODEL = "RetinaFace";
    private static final String URL = "https://example.com/retinaface.zip";
    private static final String DEST = "/tmp/models";

    private final RecordingModelDownloadListener listener =
            new RecordingModelDownloadListener();

    private DownloadProgress newProgress() {
        return new DownloadProgress(MODEL, URL, DEST, listener);
    }

    @Test
    void downloadCycleReportsStartProgressAndEnd() {
        DownloadProgress progress = newProgress();

        progress.reset("Downloading", 1000);
        assertTrue(progress.isDownloading());
        assertEquals(1, listener.startCount());
        RecordingModelDownloadListener.StartEvent start = listener.startFor(MODEL);
        assertNotNull(start);
        assertEquals(MODEL, start.modelName);
        assertEquals(URL, start.url);
        assertEquals(DEST, start.destinationDirectory);
        assertEquals(1000, start.totalBytes);

        progress.increment(250);
        progress.increment(250);
        assertEquals(500, listener.progress().get(listener.progressCount() - 1).downloadedBytes);
        assertEquals(1000, listener.progress().get(listener.progressCount() - 1).totalBytes);

        progress.end();
        assertFalse(progress.isDownloading());
        assertEquals(1, listener.downloadCount());
        assertEquals(MODEL, listener.downloaded().get(0));
    }

    @Test
    void unknownSizeIsReportedAsNegativeTotal() {
        DownloadProgress progress = newProgress();

        progress.reset("Downloading", 0);
        assertEquals(-1, listener.startFor(MODEL).totalBytes);

        progress.increment(10);
        assertEquals(-1, listener.progress().get(0).totalBytes);
    }

    @Test
    void loadingPhaseIsIgnored() {
        DownloadProgress progress = newProgress();

        progress.reset("Loading", 2);
        progress.update(1);
        progress.increment(5);
        progress.end();

        assertTrue(listener.starts().isEmpty());
        assertTrue(listener.progress().isEmpty());
        assertTrue(listener.downloaded().isEmpty());
    }

    @Test
    void progressIgnoredWhenNoDownloadStarted() {
        DownloadProgress progress = newProgress();

        progress.increment(42);
        progress.end();

        assertTrue(listener.progress().isEmpty());
        assertTrue(listener.downloaded().isEmpty());
    }

    @Test
    void endWithoutDownloadDoesNotNotify() {
        DownloadProgress progress = newProgress();
        progress.end();
        assertTrue(listener.downloaded().isEmpty());
    }

    @Test
    void constructorRejectsNullArguments() {
        assertThrows(NullPointerException.class,
                () -> new DownloadProgress(null, URL, DEST, listener));
        assertThrows(NullPointerException.class,
                () -> new DownloadProgress(MODEL, null, DEST, listener));
        assertThrows(NullPointerException.class,
                () -> new DownloadProgress(MODEL, URL, null, listener));
        assertThrows(NullPointerException.class,
                () -> new DownloadProgress(MODEL, URL, DEST, null));
    }
}