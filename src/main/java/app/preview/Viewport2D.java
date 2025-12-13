package app.preview;

import app.noisePreview.ResourceGenerationParams;
import app.noisePreview.TerrainGenerationParams;
import javafx.scene.canvas.Canvas;

/**
 * Reusable pan/zoom controller for 2D grid previews.
 * Offsets are tracked in grid-cell units (not pixels).
 */
public class Viewport2D {
    private final int cellSize;
    private final double minScale;
    private final double maxScale;

    private final TerrainGenerationParams terrainParams;
    private final ResourceGenerationParams resourceParams;

    private double offsetX;
    private double offsetY;

    private double lastMouseX;
    private double lastMouseY;

    public Viewport2D(
            int cellSize,
            double minScale,
            double maxScale,
            TerrainGenerationParams terrainParams,
            ResourceGenerationParams resourceParams
    ) {
        this.cellSize = cellSize;
        this.minScale = minScale;
        this.maxScale = maxScale;
        this.terrainParams = terrainParams;
        this.resourceParams = resourceParams;
    }

    public double getOffsetX() {
        return offsetX;
    }

    public double getOffsetY() {
        return offsetY;
    }

    public void reset() {
        offsetX = 0;
        offsetY = 0;
    }

    public void attach(Canvas canvas, Runnable onViewportChanged) {
        canvas.setOnMousePressed(e -> {
            lastMouseX = e.getX();
            lastMouseY = e.getY();
        });

        canvas.setOnMouseDragged(e -> {
            offsetX -= (e.getX() - lastMouseX) / cellSize;
            offsetY -= (e.getY() - lastMouseY) / cellSize;
            lastMouseX = e.getX();
            lastMouseY = e.getY();
            onViewportChanged.run();
        });

        canvas.setOnScroll(e -> {
            double scaleChange = e.getDeltaY() > 0 ? 1.1 : 0.909;
            double nextScale = terrainParams.scale * scaleChange;
            if ((scaleChange > 1 && nextScale > maxScale) || (scaleChange < 1 && nextScale < minScale)) {
                return;
            }
            terrainParams.scale = nextScale;
            resourceParams.scale = resourceParams.scale * scaleChange;
            onViewportChanged.run();
            e.consume();
        });
    }
}
