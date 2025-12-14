package app.psoPreview;

import java.util.List;

import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;
import swarm.Particle;

public class PsoOverlayRenderer {
    private final int gridWidth;
    private final int gridHeight;

    public PsoOverlayRenderer(int gridWidth, int gridHeight) {
        this.gridWidth = Math.max(1, gridWidth);
        this.gridHeight = Math.max(1, gridHeight);
    }

    public void render(GraphicsContext gc,
                       List<Particle> particles,
                       double[] globalBestPosition,
                       Color particleColor,
                       Color personalBestColor,
                       Color globalBestColor) {
        if (particles == null || particles.isEmpty()) {
            return;
        }

        CellLayout layout = computeLayout(gc);
        if (layout.cellSize <= 0) {
            return;
        }

        // Particles (current positions)
        gc.setFill(particleColor != null ? particleColor : Color.rgb(255, 80, 80, 0.85));
        for (Particle p : particles) {
            double[] pos = p.getPosition();
            drawDot(gc, layout, pos, 0.35);
        }

        // Personal bests
        gc.setFill(personalBestColor != null ? personalBestColor : Color.rgb(80, 160, 255, 0.65));
        for (Particle p : particles) {
            double[] pos = p.getPersonalBestPosition();
            drawDot(gc, layout, pos, 0.25);
        }

        // Global best (highlight)
        if (globalBestPosition != null && globalBestPosition.length >= 2) {
            gc.setStroke(globalBestColor != null ? globalBestColor : Color.rgb(255, 255, 255, 0.9));
            gc.setLineWidth(2.0);
            drawRing(gc, layout, globalBestPosition, 0.65);
        }
    }

    private void drawDot(GraphicsContext gc, CellLayout layout, double[] pos, double radiusCells) {
        if (pos == null || pos.length < 2) {
            return;
        }
        double px = clamp(pos[0]);
        double py = clamp(pos[1]);

        double r = radiusCells * layout.cellSize;
        double x = layout.originX + px * layout.cellSize + (layout.cellSize / 2.0) - r;
        double y = layout.originY + py * layout.cellSize + (layout.cellSize / 2.0) - r;
        gc.fillOval(x, y, r * 2, r * 2);
    }

    private void drawRing(GraphicsContext gc, CellLayout layout, double[] pos, double radiusCells) {
        double px = clamp(pos[0]);
        double py = clamp(pos[1]);

        double r = radiusCells * layout.cellSize;
        double x = layout.originX + px * layout.cellSize + (layout.cellSize / 2.0) - r;
        double y = layout.originY + py * layout.cellSize + (layout.cellSize / 2.0) - r;
        gc.strokeOval(x, y, r * 2, r * 2);
    }

    // Drawing should be robust even if the particle went out-of-bounds.
    private double clamp(double v) {
        if (Double.isNaN(v) || Double.isInfinite(v)) {
            return 0;
        }
        if (v < 0) {
            return 0;
        }
        double max = Math.max(0, gridWidth - 1);
        if (v > max) {
            return max;
        }
        return v;
    }

    private CellLayout computeLayout(GraphicsContext gc) {
        if (gc == null || gc.getCanvas() == null) {
            return new CellLayout(0, 0, 0);
        }
        double w = gc.getCanvas().getWidth();
        double h = gc.getCanvas().getHeight();
        if (w <= 0 || h <= 0) {
            return new CellLayout(0, 0, 0);
        }
        double cellSize = Math.min(w / gridWidth, h / gridHeight);
        double drawW = cellSize * gridWidth;
        double drawH = cellSize * gridHeight;
        double originX = (w - drawW) / 2.0;
        double originY = (h - drawH) / 2.0;
        return new CellLayout(originX, originY, cellSize);
    }

    private static final class CellLayout {
        private final double originX;
        private final double originY;
        private final double cellSize;

        private CellLayout(double originX, double originY, double cellSize) {
            this.originX = originX;
            this.originY = originY;
            this.cellSize = cellSize;
        }
    }
}
