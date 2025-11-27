package app;

import terrain.NoiseMapGenerator;
import terrain.TerrainAnalyzer;
import terrain.NaturalResourceRandomizer;

import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Label;
import javafx.scene.control.Slider;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.layout.BorderPane;
import javafx.scene.control.Accordion;
import javafx.scene.control.TitledPane;
import javafx.scene.paint.Color;

public class NoisePreview2D {

    private static final int WIDTH = 128;
    private static final int HEIGHT = 128;
    private static final int DISPLAY_SIZE = 512;
    private static final int CELL_SIZE = DISPLAY_SIZE / WIDTH;

    // --- NEW: movement state ---
    private double offsetX = 0;
    private double offsetY = 0;

    private double lastMouseX;
    private double lastMouseY;

    // Graphics Contexts
    private GraphicsContext noiseGC;
    private GraphicsContext terrainGC;

    // Terrain generation parameters (moved to fields)
    private double scale = 2.0;
    private int octaves = 4;
    private double persistence = 0.5;
    private double lacunarity = 2.0;
    private long seed = 42;
    private double flattenPower = 1.0;

    // Resource generation parameters (new fields)
    private double oreProbability = 0.05;
    private double fertileProbability = 0.2;
    private double forestProbability = 0.15;
    private long resourceRandomizerSeed = 12345;

    // Generated maps
    private double[][] heightMap; // Raw height values
    private int[][] terrainTypes; // Categorized terrain types
    private double[][] flatnessMap; // For resource weighting
    private int[][] resourceTypes; // Combined categorized resource types

    public Scene showNoisePreview2D() {
        // Layout setup using BorderPane
        BorderPane root = new BorderPane();
        VBox controls = new VBox(10);
        controls.setPadding(new javafx.geometry.Insets(10));
        controls.setPrefWidth(280); // Similar to Main.java

        Canvas noiseCanvas = new Canvas(DISPLAY_SIZE, DISPLAY_SIZE);
        Canvas terrainCanvas = new Canvas(DISPLAY_SIZE, DISPLAY_SIZE);

        noiseGC = noiseCanvas.getGraphicsContext2D();
        terrainGC = terrainCanvas.getGraphicsContext2D();

        HBox canvasContainer = new HBox(10, noiseCanvas, terrainCanvas);
        root.setLeft(controls);
        root.setCenter(canvasContainer);

        Scene scene = new Scene(root, DISPLAY_SIZE * 2 + 10 + controls.getPrefWidth(), DISPLAY_SIZE + 20);
        // primaryStage.setScene(scene); // Removed
        // primaryStage.setTitle("Noise and Resource Preview 2D – Movable View"); // Removed
        // primaryStage.show(); // Removed

        // --- Noise Generation Controls ---
        VBox noiseControlsVBox = new VBox(5);
        Label scaleLabel = new Label("Scale.");
        Slider scaleSlider = createSlider(0, 10, scale, 0.1);
        scaleSlider.valueProperty().addListener((obs, oldVal, newVal) -> {
            scale = newVal.doubleValue();
            updateDisplay();
        });

        Label octavesLabel = new Label("Octaves.");
        Slider octavesSlider = createSlider(1, 30, octaves, 1);
        octavesSlider.valueProperty().addListener((obs, oldVal, newVal) -> {
            octaves = newVal.intValue();
            updateDisplay();
        });

        Label persistenceLabel = new Label("Persistence.");
        Slider persistenceSlider = createSlider(0.1, 1.0, persistence, 0.05);
        persistenceSlider.valueProperty().addListener((obs, oldVal, newVal) -> {
            persistence = newVal.doubleValue();
            updateDisplay();
        });

        Label lacunarityLabel = new Label("Lacunarity.");
        Slider lacunaritySlider = createSlider(1.0, 4.0, lacunarity, 0.1);
        lacunaritySlider.valueProperty().addListener((obs, oldVal, newVal) -> {
            lacunarity = newVal.doubleValue();
            updateDisplay();
        });

        Label seedLabel = new Label("Seed.");
        Slider seedSlider = createSlider(0, 10000, seed, 1);
        seedSlider.valueProperty().addListener((obs, oldVal, newVal) -> {
            seed = newVal.longValue();
            updateDisplay();
        });

        Label flattenPowerLabel = new Label("Flatten Power.");
        Slider flattenPowerSlider = createSlider(0.1, 5.0, flattenPower, 0.1);
        flattenPowerSlider.valueProperty().addListener((obs, oldVal, newVal) -> {
            flattenPower = newVal.doubleValue();
            updateDisplay();
        });

        noiseControlsVBox.getChildren().addAll(
                scaleLabel, scaleSlider,
                octavesLabel, octavesSlider,
                persistenceLabel, persistenceSlider,
                lacunarityLabel, lacunaritySlider,
                seedLabel, seedSlider,
                flattenPowerLabel, flattenPowerSlider
        );

        TitledPane noisePane = new TitledPane("Noise Generation", noiseControlsVBox);
        Accordion accordion = new Accordion();
        accordion.getPanes().add(noisePane);
        accordion.setExpandedPane(noisePane); // Expand noise by default

        controls.getChildren().add(accordion);

        // --- Resource Generation Sliders ---
        VBox resourceControlsVBox = new VBox(5);

        Label oreProbLabel = new Label("Ore Probability.");
        Slider oreProbSlider = createSlider(0.0, 0.2, oreProbability, 0.01);
        oreProbSlider.valueProperty().addListener((obs, oldVal, newVal) -> {
            oreProbability = newVal.doubleValue();
            updateDisplay();
        });

        Label fertileProbLabel = new Label("Fertile Land Probability.");
        Slider fertileProbSlider = createSlider(0.0, 0.5, fertileProbability, 0.01);
        fertileProbSlider.valueProperty().addListener((obs, oldVal, newVal) -> {
            fertileProbability = newVal.doubleValue();
            updateDisplay();
        });

        Label forestProbLabel = new Label("Forest Probability.");
        Slider forestProbSlider = createSlider(0.0, 0.4, forestProbability, 0.01);
        forestProbSlider.valueProperty().addListener((obs, oldVal, newVal) -> {
            forestProbability = newVal.doubleValue();
            updateDisplay();
        });

        Label resourceSeedLabel = new Label("Resource Randomizer Seed.");
        Slider resourceSeedSlider = createSlider(0, 10000, resourceRandomizerSeed, 1);
        resourceSeedSlider.valueProperty().addListener((obs, oldVal, newVal) -> {
            resourceRandomizerSeed = newVal.longValue();
            updateDisplay();
        });

        resourceControlsVBox.getChildren().addAll(
                oreProbLabel, oreProbSlider,
                fertileProbLabel, fertileProbSlider,
                forestProbLabel, forestProbSlider,
                resourceSeedLabel, resourceSeedSlider
        );

        TitledPane resourcePane = new TitledPane("Resource Generation", resourceControlsVBox);
        accordion.getPanes().add(resourcePane);

        // --- MOUSE DRAGGING FOR MOVEMENT (BOTH CANVASES) ---
        noiseCanvas.setOnMousePressed(e -> {
            lastMouseX = e.getX();
            lastMouseY = e.getY();
        });

        noiseCanvas.setOnMouseDragged(e -> {
            double deltaX = e.getX() - lastMouseX;
            double deltaY = e.getY() - lastMouseY;

            offsetX -= deltaX / CELL_SIZE;
            offsetY -= deltaY / CELL_SIZE;

            lastMouseX = e.getX();
            lastMouseY = e.getY();
            updateDisplay();
        });

        terrainCanvas.setOnMousePressed(e -> {
            lastMouseX = e.getX();
            lastMouseY = e.getY();
        });

        terrainCanvas.setOnMouseDragged(e -> {
            double deltaX = e.getX() - lastMouseX;
            double deltaY = e.getY() - lastMouseY;

            offsetX -= deltaX / CELL_SIZE;
            offsetY -= deltaY / CELL_SIZE;

            lastMouseX = e.getX();
            lastMouseY = e.getY();
            updateDisplay();
        });

        updateDisplay(); // Initial draw

        return scene; // Return the scene
    }

    private Slider createSlider(double min, double max, double initial, double blockIncrement) {
        Slider slider = new Slider(min, max, initial);
        slider.setBlockIncrement(blockIncrement);
        slider.setShowTickLabels(true);
        slider.setShowTickMarks(true);
        slider.setMajorTickUnit((max - min) / 4);
        slider.setMinorTickCount(5);
        return slider;
    }

    private void generateTerrainMaps() {
        NoiseMapGenerator generator = new NoiseMapGenerator(seed);
        heightMap = generator.generateWithOffset(
                WIDTH, HEIGHT,
                scale, octaves, persistence, lacunarity, flattenPower,
                offsetX, offsetY
        );
        TerrainAnalyzer analyzer = new TerrainAnalyzer(heightMap);
        terrainTypes = analyzer.categorizeTerrain(0.3, 0.5, 0.7);
        flatnessMap = analyzer.computeFlatness();
    }

    private void generateResourceMaps() {
        NaturalResourceRandomizer resourceRandomizer = new NaturalResourceRandomizer(WIDTH, HEIGHT, resourceRandomizerSeed);

        boolean[][] oreMap = resourceRandomizer.randomizeResourceWeighted(terrainTypes, flatnessMap, "ore", oreProbability);
        boolean[][] fertileMap = resourceRandomizer.randomizeResourceWeighted(terrainTypes, flatnessMap, "fertile", fertileProbability);
        boolean[][] forestMap = resourceRandomizer.randomizeResourceWeighted(terrainTypes, flatnessMap, "forest", forestProbability);

        resourceTypes = new int[WIDTH][HEIGHT];
        for (int x = 0; x < WIDTH; x++) {
            for (int y = 0; y < HEIGHT; y++) {
                if (oreMap[x][y]) {
                    resourceTypes[x][y] = 3; // Ores (highest priority)
                } else if (forestMap[x][y]) {
                    resourceTypes[x][y] = 2; // Forest
                } else if (fertileMap[x][y]) {
                    resourceTypes[x][y] = 1; // Fertile Land
                } else {
                    resourceTypes[x][y] = 0; // Barren
                }
            }
        }
    }

    private void updateDisplay() {
        generateTerrainMaps();
        generateResourceMaps();

        // --- Draw noise ---
        for (int x = 0; x < WIDTH; x++) {
            for (int y = 0; y < HEIGHT; y++) {
                int gray = (int) (heightMap[x][y] * 255);
                gray = Math.max(0, Math.min(255, gray));
                noiseGC.setFill(Color.rgb(gray, gray, gray));
                noiseGC.fillRect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE);
            }
        }

        // --- Draw terrain and resources ---
        for (int x = 0; x < WIDTH; x++) {
            for (int y = 0; y < HEIGHT; y++) {
                Color terrainColor;
                switch (terrainTypes[x][y]) {
                    case 0 -> terrainColor = Color.BLUE;         // Water
                    case 1 -> terrainColor = Color.LIGHTGREEN;  // Plains
                    case 2 -> terrainColor = Color.DARKGREEN;   // Hill
                    case 3 -> terrainColor = Color.GRAY;        // Mountain
                    default -> terrainColor = Color.BLACK;
                }
                terrainGC.setFill(terrainColor);
                terrainGC.fillRect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE);

                // Draw resources on top of terrain
                Color resourceColor = null;
                switch (resourceTypes[x][y]) {
                    case 0: // Barren/Low
                        break;
                    case 1: // Fertile Land
                        resourceColor = Color.web("#A0F0A0", 0.6); // Lighter green for fertile, semi-transparent
                        break;
                    case 2: // Forest
                        resourceColor = Color.web("#228B22", 0.7); // Forest Green, semi-transparent
                        break;
                    case 3: // Ores
                        resourceColor = Color.web("#CD7F32", 0.8); // Bronze/Metallic, semi-transparent
                        break;
                }

                if (resourceColor != null) {
                    terrainGC.setFill(resourceColor);
                    terrainGC.fillRect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE);
                }
            }
        }
    }

    // main method is removed as this class is now a component
}
