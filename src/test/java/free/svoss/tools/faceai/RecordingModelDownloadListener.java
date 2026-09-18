package free.svoss.tools.faceai;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Test helper that records every model download event for later assertions.
 */
public final class RecordingModelDownloadListener implements ModelDownloadListener {

    /** A recorded start event. */
    public static final class StartEvent {
        public final String modelName;
        public final String url;
        public final String destinationDirectory;
        public final long totalBytes;

        StartEvent(String modelName, String url, String destinationDirectory, long totalBytes) {
            this.modelName = modelName;
            this.url = url;
            this.destinationDirectory = destinationDirectory;
            this.totalBytes = totalBytes;
        }
    }

    /** A recorded progress event. */
    public static final class ProgressEvent {
        public final String modelName;
        public final long downloadedBytes;
        public final long totalBytes;

        ProgressEvent(String modelName, long downloadedBytes, long totalBytes) {
            this.modelName = modelName;
            this.downloadedBytes = downloadedBytes;
            this.totalBytes = totalBytes;
        }
    }

    private final List<StartEvent> starts = new ArrayList<>();
    private final List<ProgressEvent> progress = new ArrayList<>();
    private final List<String> downloaded = new ArrayList<>();

    @Override
    public void onModelDownloadStarted(String modelName, String url,
            String destinationDirectory, long totalBytes) {
        starts.add(new StartEvent(modelName, url, destinationDirectory, totalBytes));
    }

    @Override
    public void onModelDownloadProgress(String modelName, long downloadedBytes, long totalBytes) {
        progress.add(new ProgressEvent(modelName, downloadedBytes, totalBytes));
    }

    @Override
    public void onModelDownloaded(String modelName) {
        downloaded.add(modelName);
    }

    public List<StartEvent> starts() {
        return starts;
    }

    public List<ProgressEvent> progress() {
        return progress;
    }

    public List<String> downloaded() {
        return downloaded;
    }

    public int startCount() {
        return starts.size();
    }

    public int progressCount() {
        return progress.size();
    }

    public int downloadCount() {
        return downloaded.size();
    }

    /**
     * Returns the recorded start event for the given model, or {@code null}.
     *
     * @param modelName the model name to look up
     * @return the matching start event, or {@code null} if none was recorded
     */
    public StartEvent startFor(String modelName) {
        for (StartEvent event : starts) {
            if (Objects.equals(event.modelName, modelName)) {
                return event;
            }
        }
        return null;
    }
}