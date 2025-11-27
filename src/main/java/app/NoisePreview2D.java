package app;

import app.noisePreview.MapGenerator;
import app.noisePreview.NoisePreview2DControls;
import app.noisePreview.ResourceGenerationParams;
import app.noisePreview.TerrainGenerationParams;
import app.noisePreview.TerrainRenderer;

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

    private double offsetX = 0;
    private double offsetY = 0;
    private double lastMouseX;
    private double lastMouseY;

    private final TerrainGenerationParams terrainParams = new TerrainGenerationParams();
    private final ResourceGenerationParams resourceParams = new ResourceGenerationParams();
    private final MapGenerator mapGenerator = new MapGenerator();
    private TerrainRenderer terrainRenderer;

    public Scene showNoisePreview2D() {
        BorderPane root = new BorderPane();
        VBox controls = new NoisePreview2DControls(terrainParams, resourceParams, () -> this.updateDisplay()).buildControlsPanel();

        Canvas noiseCanvas = new Canvas(DISPLAY_SIZE, DISPLAY_SIZE);
        Canvas terrainCanvas = new Canvas(DISPLAY_SIZE, DISPLAY_SIZE);

        GraphicsContext noiseGC = noiseCanvas.getGraphicsContext2D();
        GraphicsContext terrainGC = terrainCanvas.getGraphicsContext2D();

        terrainRenderer = new TerrainRenderer(noiseGC, terrainGC);

        HBox canvasContainer = new HBox(10, noiseCanvas, terrainCanvas);

        root.setLeft(controls);
        root.setCenter(canvasContainer);

        Scene scene = new Scene(root, DISPLAY_SIZE * 2 + 10 + controls.getPrefWidth(), DISPLAY_SIZE + 20);

        setupMouseHandlers(noiseCanvas);
        setupMouseHandlers(terrainCanvas);

        updateDisplay();
        return scene;
    }

    private void setupMouseHandlers(Canvas canvas) {
        canvas.setOnMousePressed(e -> {
            lastMouseX = e.getX();
            lastMouseY = e.getY();
        });

        canvas.setOnMouseDragged(e -> {
            offsetX -= (e.getX() - lastMouseX) / CELL_SIZE;
            offsetY -= (e.getY() - lastMouseY) / CELL_SIZE;

            lastMouseX = e.getX();
            lastMouseY = e.getY();
            updateDisplay();
        });
    }

    private void updateDisplay() {
        mapGenerator.generateTerrainMaps(terrainParams, offsetX, offsetY);
        mapGenerator.generateResourceMaps(resourceParams, offsetX, offsetY);

        terrainRenderer.renderAll(
                mapGenerator.getHeightMap(),
                mapGenerator.getTerrainTypes(),
                mapGenerator.getResourceMaps(),
                resourceParams
        );
    }
}
