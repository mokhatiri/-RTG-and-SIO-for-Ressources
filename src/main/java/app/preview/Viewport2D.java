package app.preview;

import app.noisePreview.ResourceGenerationParams;
import app.noisePreview.TerrainGenerationParams;
import javafx.scene.canvas.Canvas;

/**
 * Reusable pan/zoom controller for 2D grid previews.
 * Offsets are tracked in grid-cell units (not pixels).
 */
public class Viewport2D {
    private final int gridWidth;
    private final int gridHeight;
    private final double minScale;
    private final double maxScale;

    private final TerrainGenerationParams terrainParams;
    private final ResourceGenerationParams resourceParams;

    private double offsetX;
    private double offsetY;

    private double lastMouseX;
    private double lastMouseY;

    public Viewport2D(
            int gridWidth,
            int gridHeight,
            double minScale,
            double maxScale,
            TerrainGenerationParams terrainParams,
            ResourceGenerationParams resourceParams
    ) {
        this.gridWidth = Math.max(1, gridWidth);
        this.gridHeight = Math.max(1, gridHeight);
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
            double cellSizePx = getCellSizePx(canvas);
            if (cellSizePx <= 0) {
                return;
            }
            offsetX -= (e.getX() - lastMouseX) / cellSizePx;
            offsetY -= (e.getY() - lastMouseY) / cellSizePx;
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

    private double getCellSizePx(Canvas canvas) {
        if (canvas == null) {
            return 0;
        }
        double w = canvas.getWidth();
        double h = canvas.getHeight();
        if (w <= 0 || h <= 0) {
            return 0;
        }
        return Math.min(w / gridWidth, h / gridHeight);
    }
}
