package app;

import app.noisePreview.MapGenerator;
import app.noisePreview.NoisePreview2DControls;
import app.noisePreview.ResourceGenerationParams;
import app.noisePreview.TerrainGenerationParams;
import app.noisePreview.TerrainRenderer;
import app.preview.Viewport2D;

import javafx.scene.Scene;

import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;

import javafx.scene.layout.HBox;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.VBox;

// Main orchestrator for the 2D Noise Preview visualization
// Coordinates terrain/resource generation and rendering with interactive controls
public class NoisePreview2D {

    private static final int DISPLAY_SIZE = 512;
    private static final int CELL_SIZE = DISPLAY_SIZE / 128;
    private static final double MAX_SCALE = 10.0;
    private static final double MIN_SCALE = 0.1;

    private final TerrainGenerationParams terrainParams = new TerrainGenerationParams();
    private final ResourceGenerationParams resourceParams = new ResourceGenerationParams();
    private final MapGenerator mapGenerator = new MapGenerator();
    private TerrainRenderer terrainRenderer;
    private Viewport2D viewport;

    public Scene showNoisePreview2D() {
        BorderPane root = new BorderPane();
        VBox controls = new NoisePreview2DControls(terrainParams, resourceParams, () -> this.updateDisplay()).buildControlsPanel();

        Canvas noiseCanvas = new Canvas(DISPLAY_SIZE, DISPLAY_SIZE);
        Canvas terrainCanvas = new Canvas(DISPLAY_SIZE, DISPLAY_SIZE);

        GraphicsContext noiseGC = noiseCanvas.getGraphicsContext2D();
        GraphicsContext terrainGC = terrainCanvas.getGraphicsContext2D();

        terrainRenderer = new TerrainRenderer(noiseGC, terrainGC);

        viewport = new Viewport2D(CELL_SIZE, MIN_SCALE, MAX_SCALE, terrainParams, resourceParams);

        HBox canvasContainer = new HBox(10, noiseCanvas, terrainCanvas);

        root.setLeft(controls);
        root.setCenter(canvasContainer);

        Scene scene = new Scene(root, DISPLAY_SIZE * 2 + 10 + controls.getPrefWidth(), DISPLAY_SIZE + 20);

        viewport.attach(noiseCanvas, this::updateDisplay);
        viewport.attach(terrainCanvas, this::updateDisplay);

        updateDisplay();
        return scene;
    }

    private void updateDisplay() {
        mapGenerator.generateTerrainMaps(terrainParams, viewport.getOffsetX(), viewport.getOffsetY());
        mapGenerator.generateResourceMaps(resourceParams, viewport.getOffsetX(), viewport.getOffsetY());

        terrainRenderer.renderAll(
                mapGenerator.getHeightMap(),
                mapGenerator.getTerrainTypes(),
                mapGenerator.getResourceMaps(),
                resourceParams
        );
    }
}
