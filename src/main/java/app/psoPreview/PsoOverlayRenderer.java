package app.psoPreview;

import java.util.List;

import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;
import swarm.Particle;

public class PsoOverlayRenderer {
    private final int cellSize;

    public PsoOverlayRenderer(int cellSize) {
        this.cellSize = cellSize;
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

        // Particles (current positions)
        gc.setFill(particleColor != null ? particleColor : Color.rgb(255, 80, 80, 0.85));
        for (Particle p : particles) {
            double[] pos = p.getPosition();
            drawDot(gc, pos, 0.35);
        }

        // Personal bests
        gc.setFill(personalBestColor != null ? personalBestColor : Color.rgb(80, 160, 255, 0.65));
        for (Particle p : particles) {
            double[] pos = p.getPersonalBestPosition();
            drawDot(gc, pos, 0.25);
        }

        // Global best (highlight)
        if (globalBestPosition != null && globalBestPosition.length >= 2) {
            gc.setStroke(globalBestColor != null ? globalBestColor : Color.rgb(255, 255, 255, 0.9));
            gc.setLineWidth(2.0);
            drawRing(gc, globalBestPosition, 0.65);
        }
    }

    private void drawDot(GraphicsContext gc, double[] pos, double radiusCells) {
        if (pos == null || pos.length < 2) {
            return;
        }
        double px = clamp(pos[0]);
        double py = clamp(pos[1]);

        double r = radiusCells * cellSize;
        double x = px * cellSize + (cellSize / 2.0) - r;
        double y = py * cellSize + (cellSize / 2.0) - r;
        gc.fillOval(x, y, r * 2, r * 2);
    }

    private void drawRing(GraphicsContext gc, double[] pos, double radiusCells) {
        double px = clamp(pos[0]);
        double py = clamp(pos[1]);

        double r = radiusCells * cellSize;
        double x = px * cellSize + (cellSize / 2.0) - r;
        double y = py * cellSize + (cellSize / 2.0) - r;
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
        if (v > 127) {
            return 127;
        }
        return v;
    }
}
