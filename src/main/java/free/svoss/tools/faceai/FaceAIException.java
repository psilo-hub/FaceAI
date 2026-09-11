package free.svoss.tools.faceai;

/**
 * Runtime exception for all FaceAI library failures.
 * Wraps underlying engine (DJL) exceptions.
 */
public class FaceAIException extends RuntimeException {

    /**
     * Constructs a new FaceAIException with the specified detail message.
     *
     * @param message the detail message
     */
    public FaceAIException(String message) {
        super(message);
    }

    /**
     * Constructs a new FaceAIException with the specified detail message and cause.
     *
     * @param message the detail message
     * @param cause   the underlying cause (e.g. a DJL exception)
     */
    public FaceAIException(String message, Throwable cause) {
        super(message, cause);
    }
}
