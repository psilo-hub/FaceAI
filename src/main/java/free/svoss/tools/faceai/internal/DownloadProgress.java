package free.svoss.tools.faceai.internal;

import ai.djl.util.Progress;
import free.svoss.tools.faceai.ModelDownloadListener;

import java.util.Objects;

/**
 * A DJL {@link Progress} that translates download events into
 * {@link ModelDownloadListener} callbacks enriched with the model name, the
 * download URL and the destination directory.
 * <p>
 * DJL reports downloads via {@code reset("Downloading", total)} followed by one
 * {@link #increment} per byte read and a final {@link #end}. It also emits a
 * second, harmless {@code reset("Loading", 2)} / {@code update(1)} pair from
 * {@code BaseModelLoader.loadModel}; this adapter ignores that "Loading" phase
 * so listeners only observe real network downloads.
 * <p>
 * Instances are used from a single download thread, but every method is
 * synchronized so a listener could safely forward events concurrently.
 *
 * <p><strong>This class is internal — not part of the public API.</strong>
 */
public final class DownloadProgress implements Progress {

    private final String modelName;
    private final String modelUrl;
    private final String destinationDirectory;
    private final ModelDownloadListener listener;

    private boolean downloading;
    private long totalBytes = -1;
    private long downloadedBytes;

    /**
     * Creates a progress adapter.
     *
     * @param modelName            logical model name (for example "RetinaFace")
     * @param modelUrl             URL the model file is downloaded from
     * @param destinationDirectory directory the models are stored under
     * @param listener             listener to notify; must not be null
     */
    public DownloadProgress(String modelName, String modelUrl,
            String destinationDirectory, ModelDownloadListener listener) {
        this.modelName = Objects.requireNonNull(modelName, "modelName");
        this.modelUrl = Objects.requireNonNull(modelUrl, "modelUrl");
        this.destinationDirectory =
                Objects.requireNonNull(destinationDirectory, "destinationDirectory");
        this.listener = Objects.requireNonNull(listener, "listener");
    }

    @Override
    public synchronized void reset(String name, long size) {
        reset(name, size, null);
    }

    @Override
    public synchronized void reset(String name, long size, String url) {
        if ("Downloading".equals(name)) {
            downloading = true;
            totalBytes = size > 0 ? size : -1;
            downloadedBytes = 0;
            listener.onModelDownloadStarted(modelName, modelUrl,
                    destinationDirectory, totalBytes);
        } else {
            // "Loading" (and any other) reset marks the non-download phase.
            downloading = false;
        }
    }

    @Override
    public synchronized void start(long size) {
        if (downloading && size > 0) {
            totalBytes = size;
        }
    }

    @Override
    public synchronized void increment(long amount) {
        if (!downloading) {
            return;
        }
        downloadedBytes += amount;
        listener.onModelDownloadProgress(modelName, downloadedBytes, totalBytes);
    }

    @Override
    public synchronized void end() {
        if (!downloading) {
            return;
        }
        downloading = false;
        listener.onModelDownloaded(modelName);
    }

    @Override
    public synchronized void update(long amount) {
        update(amount, null);
    }

    @Override
    public synchronized void update(long amount, String marker) {
        // Ignored: DJL calls update(1) after the "Loading" reset in loadModel().
    }

    /**
     * Returns whether a download is currently in progress (test helper).
     *
     * @return {@code true} if a download reset has been seen but not yet ended
     */
    public synchronized boolean isDownloading() {
        return downloading;
    }
}