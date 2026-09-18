package free.svoss.tools.faceai;

/**
 * Listener for FaceAI model download progress.
 * <p>
 * Methods are invoked from the thread that triggers the download. Callers that
 * update a UI must forward the events to the UI thread (for example with
 * {@code Platform.runLater} on JavaFX).
 * <p>
 * The listener only receives events while a model is actually downloaded from
 * the network. When a model is already cached, no events are fired for it.
 *
 * <pre>{@code
 * FaceAIConfig config = FaceAIConfig.builder()
 *         .modelDownloadListener(new ModelDownloadListener() {
 *             public void onModelDownloadStarted(String modelName, String url,
 *                     String destinationDirectory, long totalBytes) { ... }
 *             public void onModelDownloadProgress(String modelName, long downloaded,
 *                     long total) { ... }
 *             public void onModelDownloaded(String modelName) { ... }
 *         })
 *         .build();
 * }</pre>
 */
public interface ModelDownloadListener {

    /**
     * Called when downloading of a model file begins.
     *
     * @param modelName            logical model name (for example "RetinaFace")
     * @param url                  URL the file is being downloaded from
     * @param destinationDirectory directory the models are stored under
     * @param totalBytes           expected total download size in bytes, or
     *                             {@code -1} if the size is unknown
     */
    void onModelDownloadStarted(String modelName, String url,
            String destinationDirectory, long totalBytes);

    /**
     * Called repeatedly while a model download makes progress.
     *
     * @param modelName       logical model name
     * @param downloadedBytes bytes downloaded so far
     * @param totalBytes      expected total download size in bytes, or
     *                        {@code -1} if the size is unknown
     */
    void onModelDownloadProgress(String modelName, long downloadedBytes, long totalBytes);

    /**
     * Called after a model finished downloading successfully.
     *
     * @param modelName logical model name
     */
    void onModelDownloaded(String modelName);
}