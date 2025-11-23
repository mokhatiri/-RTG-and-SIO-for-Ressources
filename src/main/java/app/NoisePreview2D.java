package app;

import terrain.NoiseMapGenerator;
import terrain.TerrainAnalyzer;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Label;
import javafx.scene.control.Slider;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.stage.Stage;

public class NoisePreview2D extends Application {

    private static final int WIDTH = 128;
    private static final int HEIGHT = 128;
    private static final int DISPLAY_SIZE = 512;
    private static final int CELL = DISPLAY_SIZE / WIDTH; // = 4px

    // --- NEW: movement state ---
    private double offsetX = 0;
    private double offsetY = 0;

    private double lastMouseX;
    private double lastMouseY;

    @Override
    public void start(Stage stage) {
        HBox root = new HBox(10);

        VBox controls = new VBox(10);
        Canvas noiseCanvas = new Canvas(DISPLAY_SIZE, DISPLAY_SIZE);
        Canvas terrainCanvas = new Canvas(DISPLAY_SIZE, DISPLAY_SIZE);

        GraphicsContext noiseGC = noiseCanvas.getGraphicsContext2D();
        GraphicsContext terrainGC = terrainCanvas.getGraphicsContext2D();

        Slider scaleSlider = new Slider(0, 100, 10);
        Slider octavesSlider = new Slider(1, 30, 4);
        Slider persistenceSlider = new Slider(0.1, 1.0, 0.5);
        Slider lacunaritySlider = new Slider(1.0, 4.0, 2.0);
        Slider seedSlider = new Slider(0, 10000, 42);
        Slider flattenPowerSlider = new Slider(0.1, 5.0, 1.0);

        controls.getChildren().addAll(
                new Label("Scale"), scaleSlider,
                new Label("Octaves"), octavesSlider,
                new Label("Persistence"), persistenceSlider,
                new Label("Lacunarity"), lacunaritySlider,
                new Label("Seed"), seedSlider,
                new Label("Flatten Power"), flattenPowerSlider
        );

        root.getChildren().addAll(controls, noiseCanvas, terrainCanvas);

        // Draw function (updated)
        Runnable drawNoise2D = () -> {
            int octaves = (int) octavesSlider.getValue();
            double scale = scaleSlider.getValue();
            double persistence = persistenceSlider.getValue();
            double lacunarity = lacunaritySlider.getValue();
            long seed = (long) seedSlider.getValue();
            double flattenPower = flattenPowerSlider.getValue();

            NoiseMapGenerator generator = new NoiseMapGenerator(seed);

            // --- NEW: offset scrolling support ---
            double[][] map = generator.generateWithOffset(
                    WIDTH, HEIGHT,
                    scale, octaves, persistence, lacunarity, flattenPower,
                    offsetX, offsetY
            );

            // --- Draw noise ---
            for (int x = 0; x < WIDTH; x++) {
                for (int y = 0; y < HEIGHT; y++) {
                    int gray = (int) (map[x][y] * 255);
                    gray = Math.max(0, Math.min(255, gray));
                    noiseGC.setFill(Color.rgb(gray, gray, gray));
                    noiseGC.fillRect(x * CELL, y * CELL, CELL, CELL);
                }
            }

            // --- Draw terrain ---
            TerrainAnalyzer analyzer = new TerrainAnalyzer(map);
            int[][] terrainTypes = analyzer.categorizeTerrain(0.3, 0.5, 0.7);

            for (int x = 0; x < WIDTH; x++) {
                for (int y = 0; y < HEIGHT; y++) {
                    Color color;
                    switch (terrainTypes[x][y]) {
                        case 0 -> color = Color.BLUE;
                        case 1 -> color = Color.LIGHTGREEN;
                        case 2 -> color = Color.DARKGREEN;
                        default -> color = Color.GRAY;
                    }

                    terrainGC.setFill(color);
                    terrainGC.fillRect(x * CELL, y * CELL, CELL, CELL);
                }
            }
        };

        // --- GUI slider update ---
        scaleSlider.valueProperty().addListener(e -> drawNoise2D.run());
        octavesSlider.valueProperty().addListener(e -> drawNoise2D.run());
        persistenceSlider.valueProperty().addListener(e -> drawNoise2D.run());
        lacunaritySlider.valueProperty().addListener(e -> drawNoise2D.run());
        seedSlider.valueProperty().addListener(e -> drawNoise2D.run());
        flattenPowerSlider.valueProperty().addListener(e -> drawNoise2D.run());

        // --- MOUSE DRAGGING FOR MOVEMENT (BOTH CANVASES) ---
        noiseCanvas.setOnMousePressed(e -> {
            lastMouseX = e.getX();
            lastMouseY = e.getY();
        });

        noiseCanvas.setOnMouseDragged(e -> {
            offsetX -= (e.getX() - lastMouseX) / CELL;
            offsetY -= (e.getY() - lastMouseY) / CELL;
            lastMouseX = e.getX();
            lastMouseY = e.getY();
            drawNoise2D.run();
        });

        terrainCanvas.setOnMousePressed(e -> {
            lastMouseX = e.getX();
            lastMouseY = e.getY();
        });

        terrainCanvas.setOnMouseDragged(e -> {
            offsetX -= (e.getX() - lastMouseX) / CELL;
            offsetY -= (e.getY() - lastMouseY) / CELL;
            lastMouseX = e.getX();
            lastMouseY = e.getY();
            drawNoise2D.run();
        });

        drawNoise2D.run();

        stage.setScene(new Scene(root));
        stage.setTitle("Noise vs Terrain Types (2D) – Movable View");
        stage.show();
    }

    public static void main(String[] args) {
        launch();
    }
}
