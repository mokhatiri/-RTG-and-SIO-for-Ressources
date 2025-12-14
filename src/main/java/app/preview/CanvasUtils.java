package app.preview;

import javafx.beans.binding.Bindings;
import javafx.beans.binding.DoubleBinding;
import javafx.scene.canvas.Canvas;
import javafx.scene.layout.Region;

public final class CanvasUtils {

    private CanvasUtils() {
    }

    /**
     * Binds a canvas to remain square and fit within the given container.
     * The canvas will resize as the container resizes.
     */
    public static void bindCanvasToSquare(Canvas canvas, Region container) {
        bindCanvasToSquare(canvas, container, 0.0);
    }

    public static void bindCanvasToSquare(Canvas canvas, Region container, double paddingPx) {
        DoubleBinding size = Bindings.createDoubleBinding(
                () -> {
                    double w = Math.max(0.0, container.getWidth() - paddingPx);
                    double h = Math.max(0.0, container.getHeight() - paddingPx);
                    double s = Math.min(w, h);
                    return Math.max(1.0, s);
                },
                container.widthProperty(),
                container.heightProperty()
        );

        canvas.widthProperty().bind(size);
        canvas.heightProperty().bind(size);
    }

    /**
     * Calls redraw whenever the canvas size changes.
     */
    public static void redrawOnResize(Canvas canvas, Runnable redraw) {
        canvas.widthProperty().addListener((obs, oldVal, newVal) -> {
            if (newVal != null && newVal.doubleValue() > 0) {
                redraw.run();
            }
        });
        canvas.heightProperty().addListener((obs, oldVal, newVal) -> {
            if (newVal != null && newVal.doubleValue() > 0) {
                redraw.run();
            }
        });
    }
}
