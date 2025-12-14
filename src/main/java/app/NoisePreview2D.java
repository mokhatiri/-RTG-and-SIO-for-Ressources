package app;

import app.noisePreview.MapGenerator;
import app.noisePreview.NoisePreview2DControls;
import app.noisePreview.ResourceGenerationParams;
import app.noisePreview.TerrainGenerationParams;
import app.noisePreview.TerrainRenderer;
import app.preview.CanvasUtils;
import app.preview.Viewport2D;

import javafx.scene.Scene;

import javafx.stage.Screen;
import javafx.geometry.Rectangle2D;

import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;

import javafx.scene.layout.HBox;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.Priority;
import javafx.scene.layout.StackPane;
import javafx.scene.layout.VBox;

// Main orchestrator for the 2D Noise Preview visualization
// Coordinates terrain/resource generation and rendering with interactive controls
public class NoisePreview2D {

    private static final int GRID_SIZE = 128;
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

        Canvas noiseCanvas = new Canvas(1, 1);
        Canvas terrainCanvas = new Canvas(1, 1);

        GraphicsContext noiseGC = noiseCanvas.getGraphicsContext2D();
        GraphicsContext terrainGC = terrainCanvas.getGraphicsContext2D();

        terrainRenderer = new TerrainRenderer(noiseGC, terrainGC);


        viewport = new Viewport2D(GRID_SIZE, GRID_SIZE, MIN_SCALE, MAX_SCALE, terrainParams, resourceParams);

        StackPane noisePane = new StackPane(noiseCanvas);
        StackPane terrainPane = new StackPane(terrainCanvas);
        noisePane.setMinSize(0, 0);
        terrainPane.setMinSize(0, 0);
        noisePane.setMaxSize(Double.MAX_VALUE, Double.MAX_VALUE);
        terrainPane.setMaxSize(Double.MAX_VALUE, Double.MAX_VALUE);
        HBox.setHgrow(noisePane, Priority.ALWAYS);
        HBox.setHgrow(terrainPane, Priority.ALWAYS);

        CanvasUtils.bindCanvasToSquare(noiseCanvas, noisePane);
        CanvasUtils.bindCanvasToSquare(terrainCanvas, terrainPane);
        CanvasUtils.redrawOnResize(noiseCanvas, this::updateDisplay);
        CanvasUtils.redrawOnResize(terrainCanvas, this::updateDisplay);

        HBox canvasContainer = new HBox(10, noisePane, terrainPane);

        root.setLeft(controls);
        root.setCenter(canvasContainer);

        Rectangle2D bounds = Screen.getPrimary().getVisualBounds();
        double initialW = Math.min(bounds.getWidth() * 0.95, 1400);
        double initialH = Math.min(bounds.getHeight() * 0.85, 900);
        Scene scene = new Scene(root, initialW, initialH);

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
