package app;

import app.noisePreview.MapGenerator;
import app.noisePreview.NoisePreview2DControls;
import app.noisePreview.ResourceGenerationParams;
import app.noisePreview.TerrainGenerationParams;
import app.noisePreview.TerrainRenderer;
import app.preview.Viewport2D;
import app.psoPreview.PsoParameters;
import app.psoPreview.PsoPreview2DControls;
import app.psoPreview.PsoOverlayRenderer;
import app.psoPreview.TerrainTypeReducer;

import swarm.ResourceFitness;
import swarm.FitnessAdapter;
import swarm.MixedFitnessFunction;
import swarm.SwarmOptimizer;
import swarm.TerrainFitness;

import javafx.application.Platform;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.Node;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.ScrollPane;
import javafx.stage.Stage;
import javafx.stage.Window;
import javafx.stage.WindowEvent;
import javafx.scene.input.KeyEvent;
import javafx.scene.input.MouseEvent;
import javafx.scene.input.ScrollEvent;

import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.chart.CategoryAxis;
import javafx.scene.chart.BarChart;

import javafx.animation.KeyFrame;
import javafx.animation.Timeline;
import javafx.util.Duration;

import java.util.ArrayList;

import java.util.function.Function;
import java.util.List;

public class PSOPreview {

    private static final int DISPLAY_SIZE = 512;
    private static final int GRID_SIZE = 128;
    private static final int CELL_SIZE = DISPLAY_SIZE / GRID_SIZE;
    private static final double MAX_SCALE = 10.0;
    private static final double MIN_SCALE = 0.1;

    private final TerrainGenerationParams terrainParams = new TerrainGenerationParams();
    private final ResourceGenerationParams resourceParams = new ResourceGenerationParams();
    private final PsoParameters psoParams = new PsoParameters();

    private final MapGenerator mapGenerator = new MapGenerator();
    private TerrainRenderer terrainRenderer;
    private PsoOverlayRenderer overlayRenderer;
    private Viewport2D viewport;

    private SwarmOptimizer optimizer;
    private Timeline timeline;
    private boolean running;

    private Canvas noiseCanvas;
    private GraphicsContext noiseGC;
    private Canvas terrainCanvas;
    private GraphicsContext terrainGC;

    private long iterationCount;
    private Label iterationLabel;
    private Label bestFitnessLabel;

    private Stage progressStage;

    private Function<double[], Double> mixedFitnessContinuous;
    private Function<double[], Double> resourceFitnessContinuous;
    private Function<double[], Double> terrainFitnessContinuous;

    private XYChart.Series<Number, Number> avgSeries;
    private XYChart.Series<Number, Number> diversitySeries;
    private XYChart.Series<Number, Number> resourceSeries;
    private XYChart.Series<Number, Number> terrainSeries;
    private XYChart.Series<String, Number> distributionSeries;
    private List<XYChart.Data<String, Number>> histogramBars;

    private int chartSampleStride = 1;
    private int tickCounter;
    private static final int MAX_POINTS = 600;

    private long lastChartsUpdateNs;
    private static final long CHART_UPDATE_INTERVAL_NS = 200_000_000L; // ~5 updates/sec
    private static final int MAX_STATS_SAMPLES = 250;

    public Scene showPSOPreview() {
        BorderPane root = new BorderPane();
        VBox terrainControls = new NoisePreview2DControls(terrainParams, resourceParams, this::regenerateAndReset).buildControlsPanel();
        VBox psoControls = new PsoPreview2DControls(
                psoParams,
            this::applyPsoChanges,
            this::refreshOverlayColors,
                this::toggleRun,
                this::stepOnce,
                this::regenerateAndReset
        ).buildControlsPanel();

        root.setLeft(terrainControls);
        root.setRight(psoControls);

        noiseCanvas = new Canvas(DISPLAY_SIZE, DISPLAY_SIZE);
        terrainCanvas = new Canvas(DISPLAY_SIZE, DISPLAY_SIZE);

        noiseGC = noiseCanvas.getGraphicsContext2D();
        terrainGC = terrainCanvas.getGraphicsContext2D();

        terrainRenderer = new TerrainRenderer(noiseGC, terrainGC);

        overlayRenderer = new PsoOverlayRenderer(CELL_SIZE);
        viewport = new Viewport2D(CELL_SIZE, MIN_SCALE, MAX_SCALE, terrainParams, resourceParams);

        viewport.attach(noiseCanvas, this::regenerateAndReset);
        viewport.attach(terrainCanvas, this::regenerateAndReset);

        HBox canvasContainer = new HBox(10, noiseCanvas, terrainCanvas);

        HBox hud = buildHud();
        root.setTop(hud);
        root.setCenter(canvasContainer);
        root.setBottom(buildBottomBar());

        Scene scene = new Scene(
            root,
            DISPLAY_SIZE * 2 + 10 + terrainControls.getPrefWidth() + psoControls.getPrefWidth(),
            DISPLAY_SIZE + 120
        );

        // When the user navigates away (stage switches to another scene), stop background work.
        // When navigating away (scene replaced), this root gets detached => stop background work.
        root.sceneProperty().addListener((obs, oldScene, newScene) -> {
            if (newScene == null) {
                dispose();
            }
        });

        // Also stop if the window is hidden/closed.
        scene.windowProperty().addListener((obs, oldWindow, newWindow) -> {
            if (oldWindow != null) {
                oldWindow.removeEventHandler(WindowEvent.WINDOW_HIDDEN, this::onWindowHidden);
            }
            if (newWindow != null) {
                newWindow.addEventHandler(WindowEvent.WINDOW_HIDDEN, this::onWindowHidden);
            }
        });

        // Any interaction with the whole screen pauses the simulation (including pan/zoom on canvases).
        // Exempt the "Start / Pause" button so toggling works predictably.
        installGlobalPauseFilters(scene);

        setupTimeline();
        regenerateAndReset();

        return scene;
    }

    private void dispose() {
        running = false;
        if (timeline != null) {
            timeline.stop();
            timeline = null;
        }
        if (progressStage != null) {
            try {
                progressStage.close();
            } catch (Exception ignored) {
            }
            progressStage = null;
        }
    }

    private void onWindowHidden(WindowEvent e) {
        dispose();
    }

    private HBox buildHud() {
        iterationLabel = new Label("Iter: 0");
        iterationLabel.getStyleClass().add("subtitle-label");
        bestFitnessLabel = new Label("Best fitness: —");
        bestFitnessLabel.getStyleClass().add("subtitle-label");

        HBox hud = new HBox(18, iterationLabel, bestFitnessLabel);
        hud.setPadding(new javafx.geometry.Insets(8, 10, 0, 10));
        return hud;
    }

    private HBox buildBottomBar() {
        Button toggle = new Button("Show progress");
        toggle.getStyleClass().add("launcher-button");
        toggle.setOnAction(e -> toggleProgressWindow(toggle));

        HBox bottom = new HBox(toggle);
        bottom.setPadding(new javafx.geometry.Insets(10));
        return bottom;
    }

    private void toggleProgressWindow(Button toggleButton) {
        ensureProgressStage();
        if (progressStage.isShowing()) {
            progressStage.hide();
            toggleButton.setText("Show progress");
        } else {
            progressStage.show();
            progressStage.toFront();
            toggleButton.setText("Hide progress");
            // Refresh after the stage is visible so we don't block the click handler/UI.
            Platform.runLater(() -> {
                lastChartsUpdateNs = 0L;
                updateCharts();
            });
        }
    }

    private void ensureProgressStage() {
        if (progressStage != null) {
            return;
        }
        progressStage = new Stage();
        progressStage.setTitle("PSO Progress");
        progressStage.setScene(buildProgressScene());
        progressStage.setWidth(980);
        progressStage.setHeight(720);
    }

    private Scene buildProgressScene() {
        VBox charts = new VBox(12);
        charts.setPadding(new javafx.geometry.Insets(12));

        charts.getChildren().addAll(
                buildAvgFitnessChart(),
                buildDiversityChart(),
                buildDistributionChart(),
                buildContributionChart()
        );

        ScrollPane scroll = new ScrollPane(charts);
        scroll.setFitToWidth(true);

        Scene scene = new Scene(scroll);
        try {
            scene.getStylesheets().add(getClass().getResource("/styles.css").toExternalForm());
        } catch (Exception ex) {
            System.err.println("Warning: styles.css not found in resources — using default styling.");
        }
        return scene;
    }

    private LineChart<Number, Number> buildAvgFitnessChart() {
        NumberAxis x = new NumberAxis();
        NumberAxis y = new NumberAxis();
        x.setLabel("Iteration");
        y.setLabel("Average fitness");
        LineChart<Number, Number> chart = new LineChart<>(x, y);
        chart.setCreateSymbols(false);
        chart.setAnimated(false);
        chart.setLegendVisible(false);
        chart.setTitle("Average swarm fitness vs iteration");
        avgSeries = new XYChart.Series<>();
        chart.getData().add(avgSeries);
        return chart;
    }

    private LineChart<Number, Number> buildDiversityChart() {
        NumberAxis x = new NumberAxis();
        NumberAxis y = new NumberAxis();
        x.setLabel("Iteration");
        y.setLabel("Avg distance to global best");
        LineChart<Number, Number> chart = new LineChart<>(x, y);
        chart.setCreateSymbols(false);
        chart.setAnimated(false);
        chart.setLegendVisible(false);
        chart.setTitle("Swarm diversity vs iteration");
        diversitySeries = new XYChart.Series<>();
        chart.getData().add(diversitySeries);
        return chart;
    }

    private BarChart<String, Number> buildDistributionChart() {
        CategoryAxis x = new CategoryAxis();
        NumberAxis y = new NumberAxis();
        x.setLabel("Fitness bins");
        y.setLabel("Count");
        BarChart<String, Number> chart = new BarChart<>(x, y);
        chart.setAnimated(false);
        chart.setLegendVisible(false);
        chart.setTitle("Fitness distribution (current swarm)");
        distributionSeries = new XYChart.Series<>();

        // Keep a stable set of bars; only update Y-values during the run.
        final int bins = 10;
        histogramBars = new ArrayList<>(bins);
        for (int i = 0; i < bins; i++) {
            XYChart.Data<String, Number> bar = new XYChart.Data<>("B" + i, 0);
            histogramBars.add(bar);
            distributionSeries.getData().add(bar);
        }

        chart.getData().add(distributionSeries);
        return chart;
    }

    private LineChart<Number, Number> buildContributionChart() {
        NumberAxis x = new NumberAxis();
        NumberAxis y = new NumberAxis();
        x.setLabel("Iteration");
        y.setLabel("Fitness components");
        LineChart<Number, Number> chart = new LineChart<>(x, y);
        chart.setCreateSymbols(false);
        chart.setAnimated(false);
        chart.setTitle("Global best: resource vs terrain contribution");
        resourceSeries = new XYChart.Series<>();
        resourceSeries.setName("Resource");
        terrainSeries = new XYChart.Series<>();
        terrainSeries.setName("Terrain");
        chart.getData().add(resourceSeries);
        chart.getData().add(terrainSeries);
        return chart;
    }

    private void updateHud() {
        if (iterationLabel == null || bestFitnessLabel == null) {
            return;
        }

        iterationLabel.setText("Iter: " + iterationCount);
        if (optimizer == null) {
            bestFitnessLabel.setText("Best fitness: —");
        } else {
            bestFitnessLabel.setText("Best fitness: " + String.format("%.4f", optimizer.getGlobalBestFitness()));
        }
    }

    private void installGlobalPauseFilters(Scene scene) {
        if (scene == null) {
            return;
        }

        scene.addEventFilter(MouseEvent.MOUSE_PRESSED, e -> {
            if (shouldPauseForTarget(e.getTarget())) running = false;
        });
        scene.addEventFilter(MouseEvent.MOUSE_DRAGGED, e -> {
            if (shouldPauseForTarget(e.getTarget())) running = false;
        });
        scene.addEventFilter(ScrollEvent.ANY, e -> {
            if (shouldPauseForTarget(e.getTarget())) running = false;
        });
        scene.addEventFilter(KeyEvent.ANY, e -> running = false);
    }

    private boolean shouldPauseForTarget(Object eventTarget) {
        Button b = findAncestorButton(eventTarget);
        if (b == null) {
            return true;
        }
        // Don't pre-pause before toggle handlers run.
        String text = b.getText();
        if (text == null) {
            return true;
        }
        return !("Start / Pause".equals(text) || "Show progress".equals(text) || "Hide progress".equals(text));
    }

    private Button findAncestorButton(Object eventTarget) {
        if (!(eventTarget instanceof Node node)) {
            return null;
        }
        Node current = node;
        while (current != null) {
            if (current instanceof Button button) {
                return button;
            }
            current = current.getParent();
        }
        return null;
    }

    private void setupTimeline() {
        if (timeline != null) {
            timeline.stop();
        }
        timeline = new Timeline(new KeyFrame(Duration.millis(33), e -> {
            if (!running) {
                return;
            }
            if (optimizer == null) {
                return;
            }
            int iters = Math.max(1, psoParams.iterationsPerTick);
            for (int i = 0; i < iters; i++) {
                optimizer.nextIteration();
            }
            iterationCount += iters;
            renderOverlay();
        }));
        timeline.setCycleCount(Timeline.INDEFINITE);
        timeline.play();
    }

    private void toggleRun() {
        running = !running;
    }

    private void applyPsoChanges() {
        running = false;
        if (mapGenerator.getTerrainTypes() == null || mapGenerator.getResourceMaps() == null) {
            regenerateAndReset();
            return;
        }
        optimizer = buildOptimizer();
        iterationCount = 0;
        resetCharts();
        renderOverlay();
    }

    private void refreshOverlayColors() {
        if (optimizer == null) {
            return;
        }
        renderOverlay();
    }

    private void stepOnce() {
        if (optimizer == null) {
            return;
        }
        optimizer.nextIteration();
        iterationCount += 1;
        renderOverlay();
    }

    private void regenerateAndReset() {
        if (terrainRenderer == null || viewport == null) {
            return;
        }
        // Regenerate terrain/resources
        mapGenerator.generateTerrainMaps(terrainParams, viewport.getOffsetX(), viewport.getOffsetY());
        mapGenerator.generateResourceMaps(resourceParams, viewport.getOffsetX(), viewport.getOffsetY());

        // Render base layers
        terrainRenderer.renderAll(
                mapGenerator.getHeightMap(),
                mapGenerator.getTerrainTypes(),
                mapGenerator.getResourceMaps(),
                resourceParams
        );

        // Reset PSO state on any landscape change
        optimizer = buildOptimizer();
        iterationCount = 0;
        resetCharts();
        renderOverlay();
    }

    private SwarmOptimizer buildOptimizer() {
        int[][] terrainReduced = TerrainTypeReducer.reduceTo4Types(mapGenerator.getTerrainTypes());
        int[][][] resourceInt = toIntResourceMap(mapGenerator.getResourceMaps());

        // Fitness defaults (keep simple; coefficients are already user-controlled)
        double flatnessCoeff = 0.5;

        TerrainFitness terrainFitness = new TerrainFitness(
                mapGenerator.getFlatnessMap(),
                flatnessCoeff,
                terrainReduced,
                -1.0, // water
                1.0,  // plains
                0.5,  // hills
                -0.25 // mountains
        );

        ResourceFitness resourceFitness = new ResourceFitness(
                mapGenerator.getFlatnessMap(),
                flatnessCoeff,
                resourceInt,
                0.15, // sedimentary
                1.0,  // gemstones
                0.7,  // iron
                0.5,  // coal
                0.9,  // gold
                0.6,  // wood
                0.4,  // cattle
                0.35, // wolf
                0.3   // fish
        );

        // Build continuous adapters for graphs (and local stats).
        int[] shape = new int[]{GRID_SIZE, GRID_SIZE};
        resourceFitnessContinuous = FitnessAdapter.toContinuous(resourceFitness, shape);
        terrainFitnessContinuous = FitnessAdapter.toContinuous(terrainFitness, shape);
        MixedFitnessFunction mixed = new MixedFitnessFunction(
            psoParams.resourceCoeff,
            psoParams.terrainCoeff,
            resourceFitness,
            terrainFitness
        );
        mixedFitnessContinuous = FitnessAdapter.toContinuous(mixed, shape);

        SwarmOptimizer so = new SwarmOptimizer(
                2,
                psoParams.weight,
                psoParams.cognitiveCoeff,
                psoParams.socialCoeff,
                psoParams.resourceCoeff,
                psoParams.terrainCoeff,
                resourceFitness,
                terrainFitness,
                new int[]{GRID_SIZE, GRID_SIZE}
        );

        int count = Math.max(5, psoParams.particleCount);
        for (int i = 0; i < count; i++) {
            so.addParticle(GRID_SIZE - 1, 0);
        }
        return so;
    }

    private void renderOverlay() {
        if (terrainRenderer == null || optimizer == null) {
            return;
        }
        // Re-render terrain/resources first so overlays don't smear
        terrainRenderer.renderAll(
                mapGenerator.getHeightMap(),
                mapGenerator.getTerrainTypes(),
                mapGenerator.getResourceMaps(),
                resourceParams
        );

        overlayRenderer.render(
            terrainGC,
            optimizer.getParticles(),
            optimizer.getGlobalBestPosition(),
            psoParams.particleColor,
            psoParams.personalBestColor,
            psoParams.globalBestColor
        );

        updateHud();
        updateCharts();
    }

    private void resetCharts() {
        tickCounter = 0;
        lastChartsUpdateNs = 0L;
        if (avgSeries != null) avgSeries.getData().clear();
        if (diversitySeries != null) diversitySeries.getData().clear();
        if (resourceSeries != null) resourceSeries.getData().clear();
        if (terrainSeries != null) terrainSeries.getData().clear();
        if (distributionSeries != null) distributionSeries.getData().clear();
    }

    private void updateCharts() {
        if (progressStage == null || !progressStage.isShowing()) {
            return;
        }
        if (optimizer == null || mixedFitnessContinuous == null) {
            return;
        }

        long now = System.nanoTime();
        if (lastChartsUpdateNs != 0L && (now - lastChartsUpdateNs) < CHART_UPDATE_INTERVAL_NS) {
            return;
        }
        lastChartsUpdateNs = now;

        tickCounter++;
        if (chartSampleStride < 1) chartSampleStride = 1;
        if ((tickCounter % chartSampleStride) != 0) {
            // Still update histogram occasionally when visible (keep it tied to stride as well).
            return;
        }

        List<swarm.Particle> particles = optimizer.getParticles();
        if (particles == null || particles.isEmpty()) {
            return;
        }

        double[] gb = optimizer.getGlobalBestPosition();

        int stride = Math.max(1, particles.size() / MAX_STATS_SAMPLES);
        int sampleCount = (particles.size() + stride - 1) / stride;

        double sum = 0.0;
        double distSum = 0.0;
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        double[] values = new double[sampleCount];
        int idx = 0;

        for (int i = 0; i < particles.size(); i += stride) {
            swarm.Particle p = particles.get(i);
            double[] pos = p.getPosition();
            double v = mixedFitnessContinuous.apply(pos);
            values[idx++] = v;
            sum += v;
            if (v < min) min = v;
            if (v > max) max = v;

            if (gb != null && gb.length >= 2 && pos != null && pos.length >= 2) {
                double dx = pos[0] - gb[0];
                double dy = pos[1] - gb[1];
                distSum += Math.sqrt(dx * dx + dy * dy);
            }
        }
        if (idx == 0) {
            return;
        }
        double avg = sum / idx;
        double diversity = distSum / idx;

        appendPoint(avgSeries, iterationCount, avg);
        appendPoint(diversitySeries, iterationCount, diversity);

        // Component contributions at global best
        if (gb != null && gb.length >= 2 && resourceFitnessContinuous != null && terrainFitnessContinuous != null) {
            appendPoint(resourceSeries, iterationCount, resourceFitnessContinuous.apply(gb));
            appendPoint(terrainSeries, iterationCount, terrainFitnessContinuous.apply(gb));
        }

        // Distribution histogram (from sampled values)
        updateHistogramFromValues(values, idx, min, max);
    }

    private void appendPoint(XYChart.Series<Number, Number> series, long x, double y) {
        if (series == null) {
            return;
        }
        series.getData().add(new XYChart.Data<>(x, y));
        if (series.getData().size() > MAX_POINTS) {
            series.getData().remove(0);
        }
    }

    private void updateHistogramFromValues(double[] values, int length, double min, double max) {
        if (distributionSeries == null || histogramBars == null) {
            return;
        }

        final int bins = 10;

        if (length <= 0) {
            return;
        }
        if (!Double.isFinite(min) || !Double.isFinite(max)) {
            return;
        }
        if (min == max) {
            max = min + 1e-9;
        }

        int[] counts = new int[bins];
        double width = (max - min) / bins;
        for (int i = 0; i < length; i++) {
            double v = values[i];
            int b = (int) ((v - min) / width);
            if (b < 0) b = 0;
            if (b >= bins) b = bins - 1;
            counts[b]++;
        }

        for (int b = 0; b < bins && b < histogramBars.size(); b++) {
            histogramBars.get(b).setYValue(counts[b]);
        }
    }

    private static int[][][] toIntResourceMap(boolean[][][] src) {
        int w = src.length;
        int h = src[0].length;
        int r = src[0][0].length;
        int[][][] out = new int[w][h][r];
        for (int x = 0; x < w; x++) {
            for (int y = 0; y < h; y++) {
                for (int k = 0; k < r; k++) {
                    out[x][y][k] = src[x][y][k] ? 1 : 0;
                }
            }
        }
        return out;
    }
}
